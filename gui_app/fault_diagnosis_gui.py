import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


APP_TITLE = "FNN Fault Diagnosis"
CHECKPOINT_NAME = "fnn_fault_checkpoint.pt"
SMOKE_TEST_LOG_NAME = "FNN_Fault_Diagnosis_GUI_smoke_test_error.txt"


def resource_path(file_name):
    """Return an external override or a PyInstaller-bundled resource."""
    candidates = []

    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / file_name)

    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir:
        candidates.append(Path(bundle_dir) / file_name)

    candidates.append(Path(__file__).resolve().parent / file_name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


class FNNClassifier(nn.Module):
    def __init__(self, architecture):
        super().__init__()

        layers = []
        for index in range(len(architecture) - 1):
            layers.append(nn.Linear(architecture[index], architecture[index + 1]))
            if index < len(architecture) - 2:
                layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class FaultPredictor:
    def __init__(self, checkpoint_path):
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        required_keys = {
            "model_state_dict",
            "x_mean",
            "x_scale",
            "class_ids",
        }
        missing_keys = required_keys.difference(checkpoint)
        if missing_keys:
            raise ValueError(
                "체크포인트에 필요한 항목이 없습니다: "
                + ", ".join(sorted(missing_keys))
            )

        weight_items = [
            (key, value)
            for key, value in checkpoint["model_state_dict"].items()
            if key.endswith(".weight") and value.ndim == 2
        ]
        weight_items.sort(key=lambda item: int(item[0].split(".")[-2]))
        if not weight_items:
            raise ValueError("체크포인트에서 Linear 계층 가중치를 찾지 못했습니다.")

        self.architecture = [int(weight_items[0][1].shape[1])]
        self.architecture.extend(int(value.shape[0]) for _, value in weight_items)

        for previous, current in zip(weight_items, weight_items[1:]):
            if int(previous[1].shape[0]) != int(current[1].shape[1]):
                raise ValueError("체크포인트의 Linear 계층 크기가 서로 연결되지 않습니다.")

        self.input_size = self.architecture[0]
        self.class_ids = np.asarray(checkpoint["class_ids"]).reshape(-1)
        self.x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float32).reshape(-1)
        self.x_scale = np.asarray(checkpoint["x_scale"], dtype=np.float32).reshape(-1)

        if self.input_size % 3 != 0:
            raise ValueError("모델 입력 개수는 3상으로 나눌 수 있어야 합니다.")
        if self.x_mean.size != self.input_size or self.x_scale.size != self.input_size:
            raise ValueError("체크포인트의 스케일러 크기가 모델 입력 크기와 다릅니다.")
        if self.class_ids.size != self.architecture[-1]:
            raise ValueError("체크포인트의 클래스 개수가 모델 출력 크기와 다릅니다.")
        if np.any(self.x_scale == 0):
            raise ValueError("체크포인트의 표준편차에 0이 포함되어 있습니다.")

        self.samples_per_phase = self.input_size // 3
        self.model = FNNClassifier(self.architecture)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, input_values):
        values = np.asarray(input_values, dtype=np.float32).reshape(1, -1)
        if values.shape[1] != self.input_size:
            raise ValueError(f"입력 데이터는 {self.input_size}개여야 합니다.")
        if not np.isfinite(values).all():
            raise ValueError("입력 데이터에 숫자가 아닌 값 또는 무한대가 있습니다.")

        scaled = (values - self.x_mean) / self.x_scale
        tensor = torch.from_numpy(scaled.astype(np.float32, copy=False))

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

        predicted_index = int(np.argmax(probabilities))
        predicted_class = self.class_ids[predicted_index].item()
        return predicted_class, probabilities


def load_input_file(file_path, input_size, samples_per_phase):
    try:
        values = np.genfromtxt(
            file_path,
            delimiter=",",
            dtype=np.float32,
            encoding="utf-8-sig",
        )
    except Exception as error:
        raise ValueError(f"CSV 파일을 읽을 수 없습니다.\n{error}") from error

    if values.size == 0:
        raise ValueError("CSV 파일이 비어 있습니다.")

    if values.ndim == 0:
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        values = values.reshape(1, -1)

    # Remove a text header row or an empty row/column parsed as all-NaN.
    values = values[~np.all(np.isnan(values), axis=1)]
    values = values[:, ~np.all(np.isnan(values), axis=0)]

    if values.size == 0:
        raise ValueError("CSV 파일에서 숫자 데이터를 찾지 못했습니다.")

    # Also accept one sample arranged as 200 rows x 3 columns or 3 rows x 200 columns.
    if values.shape == (samples_per_phase, 3):
        features = values.T.reshape(1, -1)
        format_message = f"{samples_per_phase}행 x 3상 형식"
    elif values.shape == (3, samples_per_phase):
        features = values.reshape(1, -1)
        format_message = f"3상 x {samples_per_phase}열 형식"
    elif values.shape[1] == input_size:
        features = values
        format_message = f"{input_size}개 입력 형식"
    elif values.shape[1] == input_size + 1:
        features = values[:, :input_size]
        format_message = "마지막 Fault ID 열 제외"
    else:
        raise ValueError(
            "지원하지 않는 CSV 구조입니다.\n"
            f"현재 구조: {values.shape[0]}행 x {values.shape[1]}열\n"
            f"지원 구조: N x {input_size}, N x {input_size + 1}, "
            f"{samples_per_phase} x 3, 3 x {samples_per_phase}"
        )

    if not np.isfinite(features).all():
        raise ValueError("입력 구간에 숫자가 아닌 값, 결측값 또는 무한대가 있습니다.")

    return features.astype(np.float32, copy=False), format_message


class FaultDiagnosisWindow(QMainWindow):
    PHASE_COLORS = ("#e53935", "#f4c20d", "#1e88e5")

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.input_rows = None
        self.current_file = None

        self.setWindowTitle(APP_TITLE)
        self.resize(1000, 760)
        self.setMinimumSize(820, 650)
        self._build_ui()
        self._apply_style()
        self._reset_plot()

    def _build_ui(self):
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)
        self.setCentralWidget(root)

        input_group = QGroupBox("입력 데이터")
        input_layout = QHBoxLayout(input_group)
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("CSV 파일을 선택하세요.")
        self.open_button = QPushButton("CSV 불러오기")
        self.open_button.clicked.connect(self.open_csv)
        self.row_label = QLabel("데이터 행")
        self.row_spin = QSpinBox()
        self.row_spin.setMinimum(1)
        self.row_spin.setMaximum(1)
        self.row_spin.setEnabled(False)
        self.row_spin.valueChanged.connect(self.run_diagnosis)
        self.diagnose_button = QPushButton("진단")
        self.diagnose_button.setEnabled(False)
        self.diagnose_button.clicked.connect(self.run_diagnosis)

        input_layout.addWidget(self.path_edit, 1)
        input_layout.addWidget(self.open_button)
        input_layout.addSpacing(8)
        input_layout.addWidget(self.row_label)
        input_layout.addWidget(self.row_spin)
        input_layout.addWidget(self.diagnose_button)
        root_layout.addWidget(input_group)

        plot_group = QGroupBox("3상 입력 플롯")
        plot_layout = QVBoxLayout(plot_group)
        self.figure = Figure(figsize=(8, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.axis = self.figure.add_subplot(111)
        plot_layout.addWidget(self.canvas)
        root_layout.addWidget(plot_group, 1)

        result_layout = QHBoxLayout()
        probability_group = QGroupBox("클래스별 진단 확률")
        probability_layout = QVBoxLayout(probability_group)
        self.probability_table = QTableWidget(len(self.predictor.class_ids), 2)
        self.probability_table.setHorizontalHeaderLabels(["Class", "Probability"])
        self.probability_table.verticalHeader().setVisible(False)
        self.probability_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.probability_table.setSelectionMode(QTableWidget.NoSelection)
        self.probability_table.horizontalHeader().setStretchLastSection(True)
        self.probability_table.setColumnWidth(0, 100)

        for row_index, class_id in enumerate(self.predictor.class_ids):
            item = QTableWidgetItem(f"Class {class_id}")
            item.setTextAlignment(Qt.AlignCenter)
            self.probability_table.setItem(row_index, 0, item)
            bar = QProgressBar()
            bar.setRange(0, 10000)
            bar.setValue(0)
            bar.setFormat("0.00%")
            self.probability_table.setCellWidget(row_index, 1, bar)

        probability_layout.addWidget(self.probability_table)

        final_group = QGroupBox("최종 진단")
        final_layout = QVBoxLayout(final_group)
        final_layout.addStretch(1)
        self.final_class_label = QLabel("-")
        self.final_class_label.setObjectName("finalClass")
        self.final_class_label.setAlignment(Qt.AlignCenter)
        self.confidence_label = QLabel("Confidence: -")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        final_layout.addWidget(self.final_class_label)
        final_layout.addWidget(self.confidence_label)
        final_layout.addStretch(1)

        result_layout.addWidget(probability_group, 3)
        result_layout.addWidget(final_group, 2)
        root_layout.addLayout(result_layout, 1)

        self.statusBar().showMessage(
            f"입력 형식: {self.predictor.input_size}열 또는 Fault ID 포함 "
            f"{self.predictor.input_size + 1}열 CSV"
        )

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow { background: #f5f6f8; }
            QGroupBox {
                background: white;
                border: 1px solid #d9dde3;
                border-radius: 7px;
                margin-top: 9px;
                padding-top: 10px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                min-height: 30px;
                padding: 0 12px;
                border: 0;
                border-radius: 5px;
                color: white;
                background: #3b6fd8;
                font-weight: 600;
            }
            QPushButton:hover { background: #315fb9; }
            QPushButton:disabled { background: #aeb7c6; }
            QLineEdit, QSpinBox {
                min-height: 28px;
                border: 1px solid #cfd5dd;
                border-radius: 4px;
                background: white;
            }
            QTableWidget {
                border: 1px solid #d9dde3;
                gridline-color: #e6e9ee;
                background: white;
            }
            QProgressBar {
                border: 1px solid #cfd5dd;
                border-radius: 3px;
                text-align: center;
                background: #eef1f5;
            }
            QProgressBar::chunk { background: #4c7de0; }
            QLabel#finalClass {
                color: #1f4fa3;
                font-size: 34px;
                font-weight: 700;
            }
            """
        )

    def _reset_plot(self):
        self.axis.clear()
        self.axis.set_xlabel("Sample")
        self.axis.set_ylabel("Phase current")
        self.axis.grid(True, alpha=0.25)
        self.canvas.draw_idle()

    def open_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "입력 데이터 선택",
            "",
            "CSV files (*.csv);;Text files (*.txt);;All files (*.*)",
        )
        if not file_path:
            return

        try:
            rows, format_message = load_input_file(
                file_path,
                self.predictor.input_size,
                self.predictor.samples_per_phase,
            )
        except Exception as error:
            QMessageBox.critical(self, "입력 오류", str(error))
            return

        self.input_rows = rows
        self.current_file = Path(file_path)
        self.path_edit.setText(file_path)
        self.row_spin.blockSignals(True)
        self.row_spin.setRange(1, len(rows))
        self.row_spin.setValue(1)
        self.row_spin.blockSignals(False)
        self.row_spin.setEnabled(len(rows) > 1)
        self.diagnose_button.setEnabled(True)
        self.statusBar().showMessage(
            f"{self.current_file.name}: {len(rows)}개 데이터 로드 완료 ({format_message})"
        )
        self.run_diagnosis()

    def run_diagnosis(self):
        if self.input_rows is None:
            return

        row_index = self.row_spin.value() - 1
        values = self.input_rows[row_index]

        try:
            predicted_class, probabilities = self.predictor.predict(values)
        except Exception as error:
            QMessageBox.critical(self, "진단 오류", str(error))
            return

        self._update_plot(values, row_index)
        self._update_probabilities(probabilities)

        confidence = float(np.max(probabilities))
        self.final_class_label.setText(f"Class {predicted_class}")
        self.confidence_label.setText(f"Confidence: {confidence * 100:.2f}%")
        self.statusBar().showMessage(
            f"데이터 {row_index + 1}/{len(self.input_rows)} 진단 완료"
        )

    def _update_plot(self, values, row_index):
        count = self.predictor.samples_per_phase
        sample_axis = np.arange(count)
        phase_values = (
            values[:count],
            values[count : 2 * count],
            values[2 * count : 3 * count],
        )

        self.axis.clear()
        for phase, color, label in zip(
            phase_values,
            self.PHASE_COLORS,
            ("A phase", "B phase", "C phase"),
        ):
            self.axis.plot(sample_axis, phase, color=color, linewidth=1.6, label=label)

        self.axis.set_title(f"Input data row {row_index + 1}")
        self.axis.set_xlabel("Sample")
        self.axis.set_ylabel("Phase current")
        self.axis.grid(True, alpha=0.25)
        self.axis.legend(loc="upper right", ncol=3)
        self.canvas.draw_idle()

    def _update_probabilities(self, probabilities):
        for row_index, probability in enumerate(probabilities):
            bar = self.probability_table.cellWidget(row_index, 1)
            percent_value = float(probability) * 100.0
            bar.setValue(round(percent_value * 100))
            bar.setFormat(f"{percent_value:.2f}%")


def main():
    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except AttributeError:
        pass

    app = QApplication(sys.argv)

    try:
        checkpoint = resource_path(CHECKPOINT_NAME)
        predictor = FaultPredictor(checkpoint)
    except Exception as error:
        if "--smoke-test" in sys.argv:
            log_path = Path(sys.executable).resolve().with_name(SMOKE_TEST_LOG_NAME)
            log_path.write_text(repr(error), encoding="utf-8")
            return 1
        QMessageBox.critical(
            None,
            "모델 로드 오류",
            f"{CHECKPOINT_NAME} 파일을 불러오지 못했습니다.\n\n{error}",
        )
        return 1

    if "--smoke-test" in sys.argv:
        try:
            predictor.predict(np.zeros(predictor.input_size, dtype=np.float32))
        except Exception as error:
            log_path = Path(sys.executable).resolve().with_name(SMOKE_TEST_LOG_NAME)
            log_path.write_text(repr(error), encoding="utf-8")
            return 1
        return 0

    window = FaultDiagnosisWindow(predictor)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
