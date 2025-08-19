import sys
import logging
from functools import partial
from pymilvus import connections, Collection
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QMessageBox

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MILVUS_HOST = "milvus-standalone"
MILVUS_PORT = "19530"
COLLECTION_NAME = "rag_demo"

class MilvusManager(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Milvus Manager - rag_demo")
        self.resize(800, 400)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Table
        self.table = QTableWidget()
        self.layout.addWidget(self.table)

        self.connect_milvus()
        self.load_data()

    def connect_milvus(self):
        log.info(f"Connexion à Milvus ({MILVUS_HOST}:{MILVUS_PORT})...")
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        self.collection = Collection(COLLECTION_NAME)
        self.collection.load()

    def load_data(self):
        log.info("Récupération des valeurs distinctes de source_path...")
        try:
            results = self.collection.query(
                expr="",  # pas de filtre
                output_fields=["source_path"],
                limit=10000  # ajustable selon le nombre max de documents
            )
        except Exception as e:
            log.error(f"Erreur lors de la récupération des données Milvus : {e}")
            QMessageBox.critical(self, "Erreur", f"Impossible de charger les données Milvus.\n{e}")
            return

        # Extraire les valeurs uniques
        pdf_paths = sorted({r.get("source_path", "") for r in results if r.get("source_path")})

        # Configurer table
        self.table.clear()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Fichier PDF", "Actions"])
        self.table.setRowCount(len(pdf_paths))

        for row, pdf_path in enumerate(pdf_paths):
            self.table.setItem(row, 0, QTableWidgetItem(pdf_path))
            btn_delete = QPushButton("Supprimer")
            btn_delete.clicked.connect(partial(self.delete_pdf, pdf_path))
            self.table.setCellWidget(row, 1, btn_delete)

    def delete_pdf(self, pdf_path):
        reply = QMessageBox.question(
            self, "Confirmation",
            f"Supprimer toutes les données liées à :\n{pdf_path} ?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            expr = f'source_path == "{pdf_path}"'
            try:
                log.info(f"Suppression des données pour {pdf_path}...")
                self.collection.delete(expr)
                self.collection.flush()
                QMessageBox.information(self, "Succès", f"Données supprimées pour {pdf_path}")
                self.load_data()
            except Exception as e:
                log.error(f"Erreur suppression {pdf_path} : {e}")
                QMessageBox.critical(self, "Erreur", f"Impossible de supprimer les données.\n{e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MilvusManager()
    window.show()
    sys.exit(app.exec_())
