#!/usr/bin/env python3
from __future__ import annotations

import colorsys
import os
import re
from typing import Callable

import pandas as pd


def run_gui(
    run_controller_fn: Callable[..., str],
    default_output_fn: Callable[[str], str],
    run_weather_fn: Callable[..., str],
    default_weather_output_fn: Callable[[str], str],
    run_weather_yearly_fn: Callable[..., dict],
    run_model_training_fn: Callable[..., dict],
    run_model_prediction_fn: Callable[..., dict],
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
        QInputDialog,
        QStackedWidget,
        QTableView,
        QVBoxLayout,
        QWidget,
    )
    from Module.Model.model_io import ensure_models_dir, load_bundle

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    # PureWeather bundles: PURE WEATHER/Models/ (copy of Module/Model/Models + new saves)
    models_dir = ensure_models_dir(os.path.join(repo_root, "PURE WEATHER", "Models"))
    parsed_weather_predictions_dir = os.path.join(
        repo_root, "PURE WEATHER", "ParsedWeatherData", "predictions"
    )
    os.makedirs(parsed_weather_predictions_dir, exist_ok=True)

    def list_csv_files(root: str) -> list[str]:
        out: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            # keep scans focused on project content
            dirnames[:] = [d for d in dirnames if d not in {".git", ".venv", "__pycache__"}]
            for name in filenames:
                if name.lower().endswith(".csv"):
                    out.append(os.path.relpath(os.path.join(dirpath, name), root))
        return sorted(out)

    def list_model_files() -> list[str]:
        if not os.path.isdir(models_dir):
            return []
        out: list[str] = []
        for name in os.listdir(models_dir):
            if name.lower().endswith(".pkl"):
                out.append(name)
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
            self.btn_run_weather = QPushButton("Weather Pipeline (column filter)")
            self.btn_run_weather_yearly = QPushButton("Weather Yearly Analytics")
            self.btn_view_csv = QPushButton("View Current CSV Files")
            self.btn_train_model = QPushButton("Train Model")
            self.btn_run_prediction = QPushButton("Run Prediction")
            self.btn_model_manager = QPushButton("Model Manager")
            menu_layout.addWidget(self.btn_run_controller)
            menu_layout.addWidget(self.btn_run_weather)
            menu_layout.addWidget(self.btn_run_weather_yearly)
            menu_layout.addWidget(self.btn_view_csv)
            menu_layout.addWidget(self.btn_train_model)
            menu_layout.addWidget(self.btn_run_prediction)
            menu_layout.addWidget(self.btn_model_manager)
            root_layout.addLayout(menu_layout)

            self.pages = QStackedWidget()
            root_layout.addWidget(self.pages)

            self.home_page = QWidget()
            home_layout = QVBoxLayout(self.home_page)
            home_label = QLabel(
                "Choose an option above.\n\n"
                "Run Controller: lake CSV — parse, lags, DO deltas.\n"
                "Weather Pipeline: raw weather CSV — four columns saved as DATE, Air temp C, RH %, pressure (default save: *_parsed.csv).\n"
                "Weather Yearly Analytics: parsed weather CSV — writes Largest_Temp_Summer, Largest_Humidity_Summer, Average_Year.\n"
                "Train Model / Run Prediction / Model Manager: models live in PURE WEATHER/Models.\n"
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

            self.model_page = QWidget()
            model_layout = QVBoxLayout(self.model_page)
            model_info = QLabel("Saved models (PURE WEATHER/Models/*.pkl):")
            self.model_list = QListWidget()
            self.model_list.setSelectionMode(QAbstractItemView.SingleSelection)
            self.model_details = QLabel("Select a model to view details.")
            self.model_details.setWordWrap(True)
            self.btn_refresh_models = QPushButton("Refresh Models")
            self.btn_delete_model = QPushButton("Delete Selected Model")
            model_layout.addWidget(model_info)
            model_layout.addWidget(self.model_list)
            model_layout.addWidget(self.model_details)
            model_buttons = QHBoxLayout()
            model_buttons.addWidget(self.btn_delete_model)
            model_buttons.addWidget(self.btn_refresh_models)
            model_layout.addLayout(model_buttons)

            self.pages.addWidget(self.home_page)
            self.pages.addWidget(self.csv_page)
            self.pages.addWidget(self.model_page)

            self.btn_run_controller.clicked.connect(self.run_controller_clicked)
            self.btn_run_weather.clicked.connect(self.run_weather_clicked)
            self.btn_run_weather_yearly.clicked.connect(self.run_weather_yearly_clicked)
            self.btn_view_csv.clicked.connect(self.show_csv_page)
            self.btn_train_model.clicked.connect(self.train_model_clicked)
            self.btn_run_prediction.clicked.connect(self.run_prediction_clicked)
            self.btn_model_manager.clicked.connect(self.show_model_page)
            self.btn_refresh_csv.clicked.connect(self.populate_csv_list)
            self.btn_open_selected_csv.clicked.connect(self.open_selected_csv)
            self.btn_refresh_models.clicked.connect(self.populate_model_list)
            self.btn_delete_model.clicked.connect(self.delete_selected_model)
            self.model_list.currentItemChanged.connect(self.show_model_details)

            self.populate_csv_list()
            self.populate_model_list()
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

        def populate_model_list(self) -> None:
            self.model_list.clear()
            models = list_model_files()
            if not models:
                self.model_list.addItem("(No model files found)")
                self.model_details.setText("No model bundles in PURE WEATHER/Models yet.")
                return
            for name in models:
                item = QListWidgetItem(name)
                item.setData(Qt.UserRole, name)
                item.setToolTip(os.path.join(models_dir, name))
                self.model_list.addItem(item)
            self.model_details.setText("Select a model to view details.")

        def show_model_page(self) -> None:
            self.populate_model_list()
            self.pages.setCurrentWidget(self.model_page)

        def show_model_details(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
            if not current:
                self.model_details.setText("Select a model to view details.")
                return
            name = current.data(Qt.UserRole)
            if not name:
                self.model_details.setText("Select a model to view details.")
                return
            model_path = os.path.join(models_dir, str(name))
            try:
                bundle = load_bundle(model_path)
                target = bundle.get("target_col", "unknown")
                mae = bundle.get("mae", "n/a")
                mae_val = bundle.get("mae_validation", "n/a")
                q_lab = bundle.get("quantile_label", "n/a")
                q_alpha = bundle.get("quantile_alpha", "n/a")
                fmode = bundle.get("feature_mode", "n/a")
                train_rows = bundle.get("train_rows", "n/a")
                val_rows = bundle.get("validation_rows", "n/a")
                test_rows = bundle.get("test_rows", "n/a")
                feat_count = len(bundle.get("feature_cols", []))
                self.model_details.setText(
                    f"Model: {os.path.basename(model_path)}\n"
                    f"Target: {target}\n"
                    f"Feature mode: {fmode}\n"
                    f"Quantile: {q_lab} (alpha={q_alpha})\n"
                    f"MAE (test): {mae}\n"
                    f"MAE (validation): {mae_val}\n"
                    f"Features: {feat_count}\n"
                    f"Train rows: {train_rows} | Val rows: {val_rows} | Test rows: {test_rows}\n"
                    f"Path: {model_path}"
                )
            except Exception as exc:
                self.model_details.setText(f"Failed to read model metadata:\n{exc}")

        def delete_selected_model(self) -> None:
            item = self.model_list.currentItem()
            if not item:
                return
            name = item.data(Qt.UserRole)
            if not name:
                return
            model_path = os.path.join(models_dir, str(name))
            if not os.path.isfile(model_path):
                self.populate_model_list()
                return

            try:
                os.remove(model_path)
            except Exception as exc:
                QMessageBox.critical(self, "Delete Error", str(exc))
                return

            self.populate_model_list()

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
            if not output_path.lower().endswith(".csv"):
                output_path = f"{output_path}.csv"

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

        def run_weather_clicked(self) -> None:
            input_path, _ = QFileDialog.getOpenFileName(
                self,
                "Choose Weather Input CSV",
                repo_root,
                "CSV Files (*.csv);;All Files (*)",
            )
            if not input_path:
                return

            default_out = default_weather_output_fn(input_path)
            output_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Weather Core CSV As",
                default_out,
                "CSV Files (*.csv);;All Files (*)",
            )
            if not output_path:
                return
            if not output_path.lower().endswith(".csv"):
                output_path = f"{output_path}.csv"

            try:
                out_path = run_weather_fn(input_path=input_path, output_path=output_path)
            except Exception as exc:
                QMessageBox.critical(self, "Weather Pipeline Error", str(exc))
                return

            QMessageBox.information(
                self,
                "Weather Pipeline Complete",
                "Columns: DATE, Air temp C, Relative Humidity (%), Atmospheric Pressure (mb), hour.\n"
                "Pressure rows containing 'm' removed; DATE rounded up to the hour; hour from DATE.\n"
                f"Saved to:\n{out_path}",
            )

        def run_weather_yearly_clicked(self) -> None:
            input_path, _ = QFileDialog.getOpenFileName(
                self,
                "Choose Parsed Weather CSV",
                repo_root,
                "CSV Files (*.csv);;All Files (*)",
            )
            if not input_path:
                return

            default_dir = os.path.dirname(input_path) or repo_root
            output_dir = QFileDialog.getExistingDirectory(
                self,
                "Choose folder for Largest_Temp_Summer.csv, Largest_Humidity_Summer.csv, Average_Year.csv",
                default_dir,
            )
            if not output_dir:
                return

            try:
                paths = run_weather_yearly_fn(input_path=input_path, output_dir=output_dir)
            except Exception as exc:
                QMessageBox.critical(self, "Weather Yearly Analytics Error", str(exc))
                return

            lines = "\n".join(f"{k}: {p}" for k, p in paths.items())
            QMessageBox.information(
                self,
                "Weather Yearly Analytics Complete",
                f"Wrote three files:\n\n{lines}",
            )

        def train_model_clicked(self) -> None:
            input_path, _ = QFileDialog.getOpenFileName(
                self,
                "Choose Training CSV",
                repo_root,
                "CSV Files (*.csv);;All Files (*)",
            )
            if not input_path:
                return

            try:
                frame = pd.read_csv(input_path, nrows=5)
            except Exception as exc:
                QMessageBox.critical(self, "CSV Read Error", str(exc))
                return
            cols = [str(c).strip() for c in frame.columns]
            target_candidates: list[str] = []
            if "DO" in cols:
                target_candidates.append("DO")
            target_candidates.extend(sorted(c for c in cols if c.startswith("DO_delta_")))
            target_candidates.extend(
                sorted(c for c in cols if re.fullmatch(r"DO_\d+h", str(c).strip()))
            )
            if not target_candidates:
                QMessageBox.warning(
                    self,
                    "No Targets Found",
                    "CSV needs a DO column and/or DO_delta_* and/or DO_*h (future level) columns.",
                )
                return

            target_col, ok = QInputDialog.getItem(
                self,
                "Select Target",
                "Train one model per run: choose DO, a single DO_delta_*, or DO_*h:",
                target_candidates,
                0,
                False,
            )
            if not ok or not target_col:
                return

            quantile_tier, ok_q = QInputDialog.getItem(
                self,
                "Quantile band",
                "Train low / mid (median) / high quantile model?",
                ["mid", "low", "high"],
                0,
                False,
            )
            if not ok_q or not quantile_tier:
                return

            default_model_path = os.path.join(
                models_dir,
                f"model_{target_col}_quantile_{quantile_tier}.pkl",
            )
            model_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Model Bundle As",
                default_model_path,
                "Pickle Files (*.pkl);;All Files (*)",
            )
            if not model_path:
                return
            if not model_path.lower().endswith(".pkl"):
                model_path = f"{model_path}.pkl"

            try:
                result = run_model_training_fn(
                    csv_path=input_path,
                    target_col=target_col,
                    model_output_path=model_path,
                    include_other_delta_features=False,
                    quantile=str(quantile_tier),
                )
            except Exception as exc:
                QMessageBox.critical(self, "Training Error", str(exc))
                return

            QMessageBox.information(
                self,
                "Training Complete",
                "Model saved.\n\n"
                f"Target: {result['target_col']}\n"
                f"Quantile: {result.get('quantile_label', '?')} (alpha={result.get('quantile_alpha', '?')})\n"
                f"MAE (test): {result['mae']:.6f}\n"
                f"MAE (validation): {result['mae_validation']:.6f}\n"
                f"Train rows: {result['train_rows']}\n"
                f"Validation rows: {result['validation_rows']}\n"
                f"Test rows: {result['test_rows']}\n"
                f"Path: {result['model_path']}",
            )

        def run_prediction_clicked(self) -> None:
            model_path, _ = QFileDialog.getOpenFileName(
                self,
                "Choose Trained Model Bundle",
                models_dir,
                "Pickle Files (*.pkl);;All Files (*)",
            )
            if not model_path:
                return

            input_path, _ = QFileDialog.getOpenFileName(
                self,
                "Choose Input CSV for Prediction",
                repo_root,
                "CSV Files (*.csv);;All Files (*)",
            )
            if not input_path:
                return

            in_base = os.path.splitext(os.path.basename(input_path))[0]
            default_pred = os.path.join(
                parsed_weather_predictions_dir,
                f"{in_base}_predictions.csv",
            )
            output_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Prediction CSV As",
                default_pred,
                "CSV Files (*.csv);;All Files (*)",
            )
            if not output_path:
                return
            if not output_path.lower().endswith(".csv"):
                output_path = f"{output_path}.csv"

            try:
                result = run_model_prediction_fn(
                    model_path=model_path,
                    input_csv_path=input_path,
                    output_csv_path=output_path,
                )
            except Exception as exc:
                QMessageBox.critical(self, "Prediction Error", str(exc))
                return

            plot_note = ""
            pp = result.get("plot_path")
            if pp:
                plot_note = f"\nPlot: {pp}"

            QMessageBox.information(
                self,
                "Prediction Complete",
                f"Rows predicted: {result['rows']}\n"
                f"Prediction column: {result['prediction_column']}\n"
                f"Saved to: {result['output_path']}{plot_note}",
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

