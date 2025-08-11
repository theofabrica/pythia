import sys
import subprocess
import threading
import requests
import time
import importlib
from importlib import import_module
from pathlib import Path
import signal
import logging
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit, QTextEdit,
    QVBoxLayout, QHBoxLayout, QComboBox, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPainter

from src.tools.chat_memory import ChatMemory
from src.tools.RAG.pdf_loader import ingest_pdf

# --------------------------------------------------------------------------- #
#                           Widgets utilitaires                               #
# --------------------------------------------------------------------------- #

class StatusLight(QWidget):
    """Petit voyant circulaire rouge / orange / vert."""
    def __init__(self):
        super().__init__()
        self.color = QColor("red")
        self.setFixedSize(20, 20)

    def set_color(self, color_name: str):
        self.color = QColor(color_name)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(self.color)
        painter.drawEllipse(0, 0, self.width(), self.height())




# --------------------------------------------------------------------------- #
#                               Interface                                     #
# --------------------------------------------------------------------------- #

class ModelChatUI(QWidget):
    """Interface principale : gestion vLLM + chains + PDF à venir."""
    append_message = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.vllm_thread = None
        self.vllm_process = None
        self.memory = ChatMemory()
        self.tei_process = None  # gestion du process TEI
        self.tei_log_thread = None  # thread de suivi des logs
        self.tei_log_process = None  # Popen docker logs
        self.current_pdf_path: str | None = None  # stocke le PDF sélectionné
        self.init_ui()


    # --------------------------- UI building -------------------------------- #

    def init_ui(self):
        project_dir = Path(__file__).resolve().parents[2]
        models_path = project_dir / "src" / "models"

        # --- Sélection du modèle ------------------------------------------------
        self.model_list = QComboBox()
        self.model_list.addItems(
            [folder.name for folder in models_path.iterdir() if folder.is_dir()]
        )

        self.launch_button = QPushButton("Lancer vLLM")
        self.launch_button.clicked.connect(self.launch_vllm)

        self.close_button = QPushButton("Fermer vLLM")
        self.close_button.setEnabled(False)
        self.close_button.clicked.connect(self.close_vllm)

        self.status_light = StatusLight()

        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Modèle :"))
        header_layout.addWidget(self.model_list)
        header_layout.addWidget(self.launch_button)
        header_layout.addWidget(self.close_button)
        header_layout.addWidget(self.status_light)

        # --- Sélection de la chain LangChain ------------------------------------
        self.chain_list = QComboBox()
        self.refresh_chains()

        chain_layout = QHBoxLayout()
        chain_layout.addWidget(QLabel("Chains :"))
        chain_layout.addWidget(self.chain_list)

        # --- Gestion RAG ---------------------------------------------------------
        self.tei_launch_button = QPushButton("Lancer RAG Tools")
        self.tei_launch_button.clicked.connect(self.launch_rag_tools)

        self.tei_close_button = QPushButton("Fermer RAG Tools")
        self.tei_close_button.setEnabled(False)
        self.tei_close_button.clicked.connect(self.close_rag_tools)

        self.tei_status_light = StatusLight()

        tei_layout = QHBoxLayout()
        tei_layout.addWidget(QLabel("RAG Tools :"))
        tei_layout.addWidget(self.tei_launch_button)
        tei_layout.addWidget(self.tei_close_button)
        tei_layout.addWidget(self.tei_status_light)

        # --- Section « Document PDF » -------------------
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Sélectionner un PDF…")

        self.browse_button = QPushButton("Parcourir")
        self.browse_button.clicked.connect(self.browse_pdf)

        self.doc_load_button = QPushButton("Charger PDF")
        self.doc_load_button.setEnabled(False)
        self.doc_load_button.clicked.connect(self.load_pdf)

        self.doc_status_light = StatusLight()  # témoin d’état du chargeur PDF

        doc_layout = QHBoxLayout()
        doc_layout.addWidget(self.file_path_edit)
        doc_layout.addWidget(self.browse_button)
        doc_layout.addWidget(self.doc_load_button)
        doc_layout.addWidget(self.doc_status_light)

        # --- Zone de chat --------------------------------------------------------
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)

        self.prompt_input = QLineEdit()
        self.send_button = QPushButton("Envoyer")
        self.send_button.clicked.connect(self.send_prompt)
        self.prompt_input.returnPressed.connect(self.send_prompt)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.prompt_input)
        input_layout.addWidget(self.send_button)

        # --- Assemblage général --------------------------------------------------
        main_layout = QVBoxLayout()
        main_layout.addWidget(QLabel("vLLM"))
        main_layout.addLayout(header_layout)
        main_layout.addWidget(QLabel("Chains"))
        main_layout.addLayout(chain_layout)
        main_layout.addWidget(QLabel("Serveur TEI"))
        main_layout.addLayout(tei_layout)
        main_layout.addWidget(QLabel("Document PDF"))
        main_layout.addLayout(doc_layout)
        main_layout.addWidget(self.chat_display)
        main_layout.addLayout(input_layout)

        self.setLayout(main_layout)
        self.setWindowTitle("Interface Chat LLM")
        self.append_message.connect(self.chat_display.append)

        # Ping serveur toutes les 2 s
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.check_server_status)
        self.status_timer.start(2000)

    # ----------------------- vLLM management ---------------------------------- #

    def vllm_server_loop(self, model_name: str):
        project_dir = Path(__file__).resolve().parents[2]
        vllm_path = project_dir / "src" / "vllm_server" / "vllm_chat_server.py"

        # NEW : start_new_session => nouveau PGID
        self.vllm_process = subprocess.Popen(
            [sys.executable, str(vllm_path), "--model_name", model_name],
            start_new_session=True          # <---
        )
        self.vllm_process.wait()

    def launch_vllm(self):
        """Thread de démarrage pour ne pas bloquer l’UI."""
        if self.vllm_thread is None or not self.vllm_thread.is_alive():
            model_name = self.model_list.currentText()
            self.vllm_thread = threading.Thread(
                target=self.vllm_server_loop, args=(model_name,), daemon=True
            )
            self.vllm_thread.start()
            self.status_light.set_color("orange")

    def close_vllm(self):
        """Stoppe tout le process-group vLLM puis flush GPU."""
        import signal, os, time, subprocess
        if self.vllm_process:
            pgid = os.getpgid(self.vllm_process.pid)
            os.killpg(pgid, signal.SIGTERM)          # SIGTERM group
            try:
                self.vllm_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)      # force kill
                self.vllm_process.wait()
            self.vllm_process = None

        # flush GPU dans un sous-processus
        project_dir = Path(__file__).resolve().parents[2]
        flush_script = project_dir / "src" / "tools" / "flush_gpu.py"
        subprocess.run([sys.executable, str(flush_script)])

        time.sleep(1)
        self.check_server_status()

        # --- Flush GPU ------------------------------------------------------
        project_dir = Path(__file__).resolve().parents[2]
        flush_script = project_dir / "src" / "tools" / "flush_gpu.py"

        if flush_script.exists():
            subprocess.run([sys.executable, str(flush_script)])
        else:
            print(f"[WARN] Script flush_gpu.py introuvable : {flush_script}")

        # --- 3) Refresh UI -----------------------------------------------------
        time.sleep(1)
        self.check_server_status()


    # --------------------------- Chains --------------------------------------- #

    def refresh_chains(self):
        """Découverte dynamique des modules dans src/chains."""
        project_dir = Path(__file__).resolve().parents[2]
        chains_path = project_dir / "src" / "chains"
        chains = [
            f.stem for f in chains_path.glob("*.py")
            if f.is_file() and not f.name.startswith("__")
        ]
        self.chain_list.clear()
        self.chain_list.addItems(chains)

    def send_prompt(self):
        """Envoie prompt → chain → affichage."""
        prompt = self.prompt_input.text().strip()
        if not prompt:
            return
        self.append_message.emit(f"[Vous] {prompt}")
        self.memory.add_user_message(prompt)
        self.prompt_input.clear()

        selected_chain = self.chain_list.currentText()
        threading.Thread(
            target=self.query_chain, args=(prompt, selected_chain), daemon=True
        ).start()

    def query_chain(self, prompt: str, chain_name: str):
        """Appel à run_chain du module sélectionné."""
        try:
            module_path = f"src.chains.{chain_name}"
            chain_module = importlib.import_module(module_path)

            project_dir = Path(__file__).resolve().parents[2]
            model_name = self.model_list.currentText()
            model_path = project_dir / "src" / "models" / model_name

            response = chain_module.run_chain(
                prompt,
                self.memory.get_history(),
                model_path=str(model_path)
            )
            self.memory.add_ai_message(response)
            self.append_message.emit(f"[LLM] {response}")
        except Exception as e:
            self.append_message.emit(f"[Erreur chain] {e}")
    # --- RAG ------------------------------------------------------
    def launch_rag_tools(self):
        """Lance les containers TEI + Milvus, puis surveille la disponibilité de Milvus."""

        def _run():
            logging.info("[RAG] Lancement des containers TEI + Milvus...")
            project_dir = Path(__file__).resolve().parents[2]
            milvus_dir = project_dir / "src" / "tools" / "RAG" / "VectorStore" / "milvus_local"
            tei_dir = project_dir / "src" / "tools" / "RAG" / "TEI"

            subprocess.run(["docker", "compose", "--env-file", ".env", "-f", "docker-compose.yml", "up", "-d"],
                           cwd=milvus_dir)
            subprocess.run(["docker", "compose", "up", "-d"], cwd=tei_dir)

            if self.tei_log_thread is None or not self.tei_log_thread.is_alive():
                self.tei_log_thread = threading.Thread(target=self.stream_tei_logs, daemon=True)
                self.tei_log_thread.start()

            # Attente fixe (modèle TEI très long à charger)
            def delayed_check():
                logging.info("[RAG] Attente du chargement initial (TEI + Milvus)...")
                time.sleep(60)
                self.wait_for_milvus_ready_background()

            threading.Thread(target=delayed_check, daemon=True).start()

        threading.Thread(target=_run, daemon=True).start()
        self.tei_status_light.set_color("orange")

    def wait_for_milvus_ready_background(self, max_wait=120, interval=5):
        """Vérifie si Milvus est prêt, dans un thread séparé (après délai de chauffe)."""

        start_time = time.time()
        while (time.time() - start_time) < max_wait:
            try:
                r = requests.get("http://milvus-standalone:19530", timeout=1)
                if r.status_code == 200:
                    logging.info("[RAG] Milvus est prêt.")
                    self.tei_status_light.set_color("green")
                    self.tei_close_button.setEnabled(True)
                    return
            except Exception:
                pass
            logging.info("[RAG] Attente de Milvus...")
            time.sleep(interval)

        logging.error("[RAG] Milvus n'a pas répondu après le délai imparti.")

    def close_rag_tools(self):
        """Arrête les containers TEI + Milvus et coupe les logs."""
        def _stop():
            project_dir = Path(__file__).resolve().parents[2]
            milvus_dir = project_dir / "src" / "tools" / "RAG" / "VectorStore" / "milvus_local"
            tei_dir = project_dir / "src" / "tools" / "RAG" / "TEI"

            subprocess.run(["docker", "compose", "down"], cwd=tei_dir)
            subprocess.run(["docker", "compose", "--env-file", ".env", "-f", "docker-compose.yml", "down"],
                           cwd=milvus_dir)

            if self.tei_log_process and self.tei_log_process.poll() is None:
                self.tei_log_process.terminate()
                self.tei_log_process.wait(timeout=3)

        threading.Thread(target=_stop, daemon=True).start()
        self.tei_status_light.set_color("red")
        self.tei_close_button.setEnabled(False)

    def stream_tei_logs(self):
        self.tei_log_process = subprocess.Popen(
            ["docker", "logs", "-f", "tei"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in self.tei_log_process.stdout:
            print(f"[TEI] {line.rstrip()}")

    # ----------------------- Server health polling ---------------------------- #

    def check_server_status(self):
        """Met à jour le voyant et l’état du bouton « Fermer »."""
        def ping():
            try:
                r = requests.get("http://localhost:8000/v1/status", timeout=1)
                if r.status_code == 200 and r.json().get("ready") is True:
                    self.status_light.set_color("green")
                    self.close_button.setEnabled(True)
                else:
                    self.status_light.set_color("orange")
                    self.close_button.setEnabled(False)
            except Exception:
                self.status_light.set_color("red")
                self.close_button.setEnabled(False)

        threading.Thread(target=ping, daemon=True).start()

        def ping_tei():
            try:
                r = requests.get("http://tei:80/health", timeout=1)
                if r.status_code == 200:  # 200 suffit
                    self.tei_status_light.set_color("green")
                    self.tei_close_button.setEnabled(True)
                else:
                    self.tei_status_light.set_color("orange")
                    self.tei_close_button.setEnabled(False)
            except Exception:
                self.tei_status_light.set_color("red")
                self.tei_close_button.setEnabled(False)

        threading.Thread(target=ping_tei, daemon=True).start()
    # --------------------------- PDF helpers ---------------------------------- #

    def browse_pdf(self):
        """Sélectionne un PDF depuis le disque."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Choisir un PDF", str(Path.home()), "PDF (*.pdf)"
        )
        if path:
            self.file_path_edit.setText(path)
            self.doc_load_button.setEnabled(True)

    def load_pdf(self):
        """Charge le PDF sélectionné via pdf_loader."""
        self.current_pdf_path = self.file_path_edit.text().strip()
        if not self.current_pdf_path:
            return

        # Appel au loader externe
        try:
            from src.tools.RAG.pdf_loader import ingest_pdf
            ingest_pdf(self.current_pdf_path)
            self.doc_status_light.set_color("green")
            self.append_message.emit(f"[RAG] PDF indexé avec succès : {Path(self.current_pdf_path).name}")
            self.doc_load_button.setEnabled(False)
        except Exception as e:
            self.doc_status_light.set_color("red")
            self.append_message.emit(f"[Erreur PDF] {e}")

    # ------------------------- Fermeture clean -------------------------------- #

    def closeEvent(self, event):
        """On ferme l’appli : stoppe vLLM et TEI."""
        self.close_vllm()
        self.close_tei()
        if self.vllm_thread:
            self.vllm_thread.join(timeout=5)
        event.accept()


# --------------------------------------------------------------------------- #
#                                Entrée main                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModelChatUI()
    window.show()
    sys.exit(app.exec_())
