#!/usr/bin/env python3
from __future__ import annotations

import colorsys
import os
from typing import Callable

import pandas as pd


def run_gui(
    run_controller_fn: Callable[..., str],
    default_output_fn: Callable[[str], str],
) -> int:
    from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QAbstractItemView,
        QStackedWidget,
        QTableView,
        QVBoxLayout,
        QWidget,
    )

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    def list_csv_files(root: str) -> list[str]:
        out: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            # keep scans focused on project content
            dirnames[:] = [d for d in dirnames if d not in {".git", ".venv", "__pycache__"}]
            for name in filenames:
                if name.lower().endswith(".csv"):
                    out.append(os.path.relpath(os.path.join(dirpath, name), root))
        return sorted(out)

    class CsvTableModel(QAbstractTableModel):
        def __init__(self, frame: pd.DataFrame) -> None:
            super().__init__()
            # Keep only display-safe objects so Qt can render reliably.
            self._df = frame.fillna("NaN").astype(object)
            self._column_colors = self._build_column_colors(len(self._df.columns))

        @staticmethod
        def _build_column_colors(num_cols: int) -> list[QColor]:
            colors: list[QColor] = []
            if num_cols <= 0:
                return colors
            # Distribute hues around the color wheel with low saturation/dark value
            # so the table remains readable while each column is visually distinct.
            for i in range(num_cols):
                h = (i / num_cols) % 1.0
                r, g, b = colorsys.hsv_to_rgb(h, 0.35, 0.27)
                colors.append(QColor(int(r * 255), int(g * 255), int(b * 255)))
            return colors

        def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
            if parent.isValid():
                return 0
            return len(self._df.index)

        def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
            if parent.isValid():
                return 0
            return len(self._df.columns)

        def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
            if not index.isValid():
                return None

            value = self._df.iat[index.row(), index.column()]

            if role == Qt.DisplayRole:
                return str(value)

            if role == Qt.TextAlignmentRole:
                col_name = str(self._df.columns[index.column()]).lower()
                if "date" in col_name:
                    return int(Qt.AlignLeft | Qt.AlignVCenter)
                return int(Qt.AlignRight | Qt.AlignVCenter)

            if role == Qt.BackgroundRole:
                v = str(value).strip().lower()
                if v == "nan" or v == "":
                    return QColor("#3B2323")  # highlight missing values

                base = self._column_colors[index.column()]
                # Slightly alternate shade by row for better scanability.
                if index.row() % 2:
                    return base
                return base.lighter(112)

            if role == Qt.ForegroundRole:
                v = str(value).strip().lower()
                if v == "nan" or v == "":
                    return QColor("#FFB4B4")
                return QColor("#EDEDED")

            return None

        def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
            if role == Qt.DisplayRole:
                if orientation == Qt.Horizontal:
                    return str(self._df.columns[section])
                return str(section + 1)
            if role == Qt.BackgroundRole and orientation == Qt.Horizontal:
                return QColor("#0F2D45")
            if role == Qt.ForegroundRole and orientation == Qt.Horizontal:
                return QColor("#F1F5F9")
            return None

    class CsvViewerWindow(QMainWindow):
        def __init__(self, csv_abs_path: str, repo_root_path: str) -> None:
            super().__init__()
            rel = os.path.relpath(csv_abs_path, repo_root_path)
            self.setWindowTitle(f"CSV Viewer - {rel}")
            self.resize(1200, 760)

            # Read as strings so raw source values (including "NaN") are preserved for viewing.
            frame = pd.read_csv(csv_abs_path, dtype=str, keep_default_na=False)
            self.model = CsvTableModel(frame)

            table = QTableView()
            table.setModel(self.model)
            table.setAlternatingRowColors(True)
            table.setSortingEnabled(True)
            table.horizontalHeader().setStretchLastSection(True)
            table.verticalHeader().setDefaultSectionSize(22)

            info = QLabel(
                f"File: {rel} | Rows: {len(frame):,} | Columns: {len(frame.columns):,}"
            )
            info.setStyleSheet("padding: 6px;")

            box = QWidget()
            layout = QVBoxLayout(box)
            layout.addWidget(info)
            layout.addWidget(table)
            self.setCentralWidget(box)

    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Lake Project - Core Menu")
            self.resize(860, 540)
            self.viewer_windows: list[QMainWindow] = []

            container = QWidget()
            self.setCentralWidget(container)
            root_layout = QVBoxLayout(container)

            header = QLabel("Core Menu")
            header.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            header.setStyleSheet("font-size: 20px; font-weight: 600;")
            root_layout.addWidget(header)

            menu_layout = QHBoxLayout()
            self.btn_run_controller = QPushButton("Run Controller (Parse + Features)")
            self.btn_view_csv = QPushButton("View Current CSV Files")
            menu_layout.addWidget(self.btn_run_controller)
            menu_layout.addWidget(self.btn_view_csv)
            root_layout.addLayout(menu_layout)

            self.pages = QStackedWidget()
            root_layout.addWidget(self.pages)

            self.home_page = QWidget()
            home_layout = QVBoxLayout(self.home_page)
            home_label = QLabel(
                "Choose an option above.\n\n"
                "Run Controller: choose an input CSV and produce the processed output.\n"
                "View Current CSV Files: list CSV files found in this project."
            )
            home_label.setWordWrap(True)
            home_layout.addWidget(home_label)
            home_layout.addStretch(1)

            self.csv_page = QWidget()
            csv_layout = QVBoxLayout(self.csv_page)
            csv_info = QLabel("Current CSV files in project:")
            self.csv_list = QListWidget()
            self.csv_list.setSelectionMode(QAbstractItemView.SingleSelection)
            self.btn_refresh_csv = QPushButton("Refresh CSV List")
            self.btn_open_selected_csv = QPushButton("Open Selected CSV")
            csv_layout.addWidget(csv_info)
            csv_layout.addWidget(self.csv_list)
            csv_buttons = QHBoxLayout()
            csv_buttons.addWidget(self.btn_open_selected_csv)
            csv_buttons.addWidget(self.btn_refresh_csv)
            csv_layout.addLayout(csv_buttons)

            self.pages.addWidget(self.home_page)
            self.pages.addWidget(self.csv_page)

            self.btn_run_controller.clicked.connect(self.run_controller_clicked)
            self.btn_view_csv.clicked.connect(self.show_csv_page)
            self.btn_refresh_csv.clicked.connect(self.populate_csv_list)
            self.btn_open_selected_csv.clicked.connect(self.open_selected_csv)

            self.populate_csv_list()
            self.pages.setCurrentWidget(self.home_page)

        def populate_csv_list(self) -> None:
            self.csv_list.clear()
            files = list_csv_files(repo_root)
            if not files:
                self.csv_list.addItem("(No CSV files found)")
                return
            for rel in files:
                item = QListWidgetItem()
                item.setData(Qt.UserRole, rel)

                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(8, 2, 8, 2)
                row_layout.setSpacing(8)

                name = QLabel(os.path.basename(rel))
                name.setStyleSheet("padding: 2px 0;")
                name.setToolTip(rel)
                open_btn = QPushButton("↗")
                open_btn.setToolTip("Open CSV in a new window")
                open_btn.setFixedWidth(34)
                open_btn.clicked.connect(lambda _=False, p=rel: self.open_csv_viewer_by_rel(p))

                row_layout.addWidget(name, 1)
                row_layout.addWidget(open_btn, 0)

                item.setSizeHint(row_widget.sizeHint())
                self.csv_list.addItem(item)
                self.csv_list.setItemWidget(item, row_widget)

        def show_csv_page(self) -> None:
            self.populate_csv_list()
            self.pages.setCurrentWidget(self.csv_page)

        def run_controller_clicked(self) -> None:
            input_path, _ = QFileDialog.getOpenFileName(
                self,
                "Choose Input CSV",
                repo_root,
                "CSV Files (*.csv);;All Files (*)",
            )
            if not input_path:
                return

            default_out = default_output_fn(input_path)
            output_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Output CSV As",
                default_out,
                "CSV Files (*.csv);;All Files (*)",
            )
            if not output_path:
                return

            try:
                out_path = run_controller_fn(input_path=input_path, output_path=output_path)
            except Exception as exc:
                QMessageBox.critical(self, "Controller Error", str(exc))
                return

            QMessageBox.information(
                self,
                "Controller Complete",
                f"Processed file saved to:\n{out_path}",
            )

        def open_selected_csv(self) -> None:
            item = self.csv_list.currentItem()
            if not item:
                QMessageBox.information(self, "No Selection", "Select a CSV file first.")
                return
            rel = item.data(Qt.UserRole)
            if not rel:
                return
            self.open_csv_viewer_by_rel(str(rel))

        def open_csv_viewer_by_rel(self, rel: str) -> None:
            csv_abs = os.path.join(repo_root, rel)
            if not os.path.isfile(csv_abs):
                QMessageBox.warning(self, "Missing File", f"File not found:\n{csv_abs}")
                return

            try:
                viewer = CsvViewerWindow(csv_abs, repo_root)
            except Exception as exc:
                QMessageBox.critical(self, "CSV Open Error", str(exc))
                return

            # Keep a reference so the child window is not garbage-collected.
            self.viewer_windows.append(viewer)
            viewer.show()

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.show()
    return app.exec()

