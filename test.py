import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import cv2
import pandas as pd
import pyttsx3  # TTS 라이브러리
from torchvision import models, transforms
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import messagebox

# YOLOv5 모델 로딩 (사람 탐지용)
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 사전 훈련된 ResNet 모델 로딩 (안경 착용 여부 분류용)
model_glasses = models.resnet18(pretrained=True)
model_glasses.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 사용자 정보 입력 받기
def get_user_input():
    user_name = input("이름을 입력하세요: ")
    user_birth = input("생년월일을 입력하세요 (YYYY-MM-DD): ")
    return user_name, user_birth

# CSV 파일 초기화
columns = ['Name', 'Birth Date', 'Glasses', 'Confidence', 'Class', 'Class Name']
df = pd.DataFrame(columns=columns)

# 예시로 안경 착용 여부를 분류하는 함수
def predict_glasses(person_image):
    image_tensor = transform(person_image)
    image_tensor = image_tensor.unsqueeze(0)  # 배치 차원 추가

    with torch.no_grad():
        output = model_glasses(image_tensor)

    _, predicted = torch.max(output, 1)
    return predicted.item()  # 0: 안경 없음, 1: 안경 있음

# tkinter를 사용한 사용자 인터페이스
def ask_user_info(user_name, user_birth):
    def on_yes_click():
        nonlocal user_confirmed
        user_confirmed = True
        root.quit()  # 이벤트 루프 종료
        root.destroy()  # 창 닫기

    def on_no_click():
        nonlocal user_confirmed
        user_confirmed = False
        root.quit()  # 이벤트 루프 종료
        root.destroy()  # 창 닫기

    def on_finish_click():
        nonlocal user_confirmed
        user_confirmed = "finish"
        root.quit()  # 이벤트 루프 종료
        root.destroy()  # 창 닫기

    user_confirmed = None
    root = tk.Tk()
    root.title("사용자 확인")
    message = f"입력한 이름: {user_name}\n입력한 생년월일: {user_birth}\n이 정보가 맞습니까?"
    label = tk.Label(root, text=message)
    label.pack()

    # 버튼 생성
    yes_button = tk.Button(root, text="Yes", command=on_yes_click)
    yes_button.pack(side=tk.LEFT, padx=10)

    no_button = tk.Button(root, text="No", command=on_no_click)
    no_button.pack(side=tk.LEFT, padx=10)

    finish_button = tk.Button(root, text="Finish", command=on_finish_click)
    finish_button.pack(side=tk.LEFT, padx=10)

    root.mainloop()  # 이벤트 루프 시작

    return user_confirmed

# 웹캡처 및 사람 탐지
def capture_and_detect_person(cap, user_name, user_birth):
    ret, frame = cap.read()
    if not ret:
        print("웹캠을 열 수 없습니다.")
        return None

    # YOLOv5 모델을 사용하여 객체 탐지
    results = model_yolo(frame)

    # 사람에 대한 탐지 결과만 추출
    detections = results.xyxy[0].cpu().numpy()
    for detection in detections:
        xmin, ymin, xmax, ymax, confidence, class_id = detection[:6]
        if int(class_id) == 0:  # class_id == 0 은 'person'
            # 사람 영역을 자르기
            person_image = frame[int(ymin):int(ymax), int(xmin):int(xmax)]

            # 안경 착용 여부 예측
            glasses_pred = predict_glasses(person_image)

            # 예측된 결과에 따라 텍스트 표시
            label = 'Glasses' if glasses_pred == 1 else 'No Glasses'
            cv2.putText(frame, label, (int(xmin), int(ymin)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            # 사람 영역에 바운딩 박스를 그리기
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)

            # 탐지된 정보를 CSV에 저장
            new_data = pd.DataFrame([{
                'Name': user_name,
                'Birth Date': user_birth,
                'Glasses': label,
                'Confidence': confidence,
                'Class': class_id,
                'Class Name': model_yolo.names[int(class_id)]
            }])

            return frame, new_data  # 결과와 탐지된 정보를 반환
    return frame, None

# 메인 프로그램
def main():
    cap = cv2.VideoCapture(0)

    while True:
        user_name, user_birth = get_user_input()

        while True:
            frame, new_data = capture_and_detect_person(cap, user_name, user_birth)
            if frame is None:
                break

            # 결과 이미지 표시
            cv2.imshow('YOLOv5 Detection', frame)

            # 'q' 키를 누르면 종료
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 종료
                break

            # 'y' 키를 누르면 새로운 사람을 입력받음
            elif key == ord('y'):  # 'y'를 누르면 새로운 사람을 입력받음
                print("이름과 생년월일을 저장 중입니다...")
                break

            if new_data is not None:
                # 기존 데이터와 합치기
                global df
                df = pd.concat([df, new_data], ignore_index=True)

        # 'Confidence' 값이 가장 높은 탐지만 남기기
        df_max_confidence = df.loc[df.groupby(['Name', 'Birth Date'])['Confidence'].idxmax()]

        # 결과를 엑셀 파일로 저장
        df_max_confidence.to_excel('detections.xlsx', index=False)

        # 사용자 확인 창 띄우기
        user_confirmed = ask_user_info(user_name, user_birth)

        if user_confirmed == True:
            print("사용자 정보 확인 완료")
            cap.release()
            cv2.destroyAllWindows()
        elif user_confirmed == False:
            print("사용자 정보를 다시 입력합니다.")
            continue
        elif user_confirmed == "finish":
            print("프로그램을 종료합니다.")
            break

        # 새로운 사람 정보 입력을 받기 위해 웹캡을 다시 시작
        cap = cv2.VideoCapture(0)

    # 웹캠 종료 및 창 닫기
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
