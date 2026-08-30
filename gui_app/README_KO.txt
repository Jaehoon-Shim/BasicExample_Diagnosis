FNN Fault Diagnosis GUI 사용 방법
=================================

1. 이 폴더에 아래 3개 파일이 함께 있는지 확인합니다.
   - fault_diagnosis_gui.py
   - fnn_fault_checkpoint.pt
   - build_windows_app.bat

2. 인터넷에 연결된 Windows PC에서 build_windows_app.bat를 실행합니다.
   처음 한 번은 PyTorch 등 빌드 패키지를 받아야 하므로 시간이 오래 걸릴 수 있습니다.

3. 빌드가 끝나면 아래 실행 파일이 생성됩니다.
   dist\FNN_Fault_Diagnosis_GUI.exe

4. 실행 파일에서 CSV 불러오기를 누르고 진단할 데이터를 선택합니다.

지원하는 입력 CSV 구조
----------------------
- N행 x 600열: A상 200개 + B상 200개 + C상 200개
- N행 x 601열: 위 600개 입력 + 마지막 Fault ID 열(진단 시 마지막 열은 제외)
- 200행 x 3열: A상, B상, C상 열로 구성된 단일 데이터
- 3행 x 200열: A상, B상, C상 행으로 구성된 단일 데이터

여러 행이 들어 있는 CSV는 GUI의 '데이터 행'에서 진단할 행을 선택할 수 있습니다.
BAT는 현재 Windows Python으로 로컬 .venv_gui_build 환경을 만들고,
그 안에 CPU PyTorch, PyQt5, Matplotlib, PyInstaller만 설치합니다.
참고 노트북과 같은 --collect-all torch 방식으로 PyTorch를 onefile EXE에 포함합니다.
원본 .pt도 같은 EXE 내부에 포함합니다.
GUI는 EXE 내부의 .pt를 torch.load로 읽고 PyTorch로 직접 추론합니다.
빌드 마지막에는 생성된 EXE가 .pt를 로드하고 추론하는 자체 진단을 수행합니다.
완성된 EXE 파일 하나만 다른 64비트 Windows PC로 옮겨 실행하면 됩니다.
