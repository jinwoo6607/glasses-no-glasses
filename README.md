웹캠으로 안경을쓴사람과 안쓴사람을 구분하고 사용자에게 입력받은 데이터들을 엑셀로 저장하는 프로그램입니다.




<img width="473" alt="1" src="https://github.com/user-attachments/assets/002e7bc3-ef9b-4ce1-b116-7974d3599445">
<img width="476" alt="사진3" src="https://github.com/user-attachments/assets/02f1f111-c808-4415-81c1-4cf93158097e">
<img width="502" alt="사진 5" src="https://github.com/user-attachments/assets/134b6c64-4c24-43aa-9527-de2f34f8619f">




Y키를 누르고



<img width="151" alt="사진6" src="https://github.com/user-attachments/assets/f6fe817a-bcd8-4087-bf10-fd0553a61a93">







정보가 맞으면 YES 키를 눌러 엑셀에 정보를 저장되며 이어서 그 다음사람의 정보를 받을수있음 
<img width="473" alt="사진7" src="https://github.com/user-attachments/assets/ca704602-5166-459f-8d1a-a505570c13f1">




정보에 수정이 필요하면 NO버튼을 눌러서 수정가능

<img width="474" alt="사진9" src="https://github.com/user-attachments/assets/270cc420-9d36-48af-b5c1-471b7cf90f32">

<img width="475" alt="1kfskd" src="https://github.com/user-attachments/assets/35e7a552-a3ec-40da-9ad4-04fc09b27ed7">




이 프로그램은는 사람을 탐지하고, 해당 사람의 얼굴에서 안경 착용 여부를 분류하는 프로그램입니다.

주요 기능:
사람 탐지: YOLOv5 모델을 이용하여 웹캠에서 사람을 탐지합니다.
안경 착용 여부 판별: ResNet18 모델을 이용하여 사람의 얼굴에서 안경 착용 여부를 판별합니다.
사용자 정보 확인: 사용자 이름과 생년월일을 입력받고, 확인 창을 통해 정보를 검증합니다.
결과 저장: 탐지된 사람 정보는 CSV 형식으로 저장되고, detections.xlsx로 엑셀 파일로 저장됩니다.
실시간 캡처: 실시간으로 사람을 캡처하여, 'y' 키를 누르면 캡쳐된 사람을 저장하게되되며 

실행 흐름:

사용자 이름과 생년월일 입력:

첫 번째로 사용자에게 이름과 생년월일을 입력받습니다.

실시간 사람 탐지:

웹캠을 통해 실시간으로 사람을 탐지합니다.
탐지된 사람에 대해 안경 착용 여부를 판별합니다.

탐지된 결과 표시:

웹캠 창에 사람을 탐지한 위치와 안경 여부를 텍스트로 표시합니다.

결과 저장:

탐지된 정보가 detections.xlsx 파일에 저장됩니다.

사용자 확인:

사용자에게 정보를 확인하는 창을 띄워, 입력된 정보가 맞는지 다시 한번 확인합니다.
'Yes'를 선택하면 입력받은 내용그대로 .xlsx에 저장되며 다음 사람 입력을 받을수있게진행되고
'No'를 선택하면 사용자 정보를 다시 입력받습니다.
'Finish'를 선택하면 마지막으로 사용자에게 입력받은 사용자 정보까지 저장하고 프로그램이 종료됩니다.

설치해야 하는 라이브러리:사용법.txt
