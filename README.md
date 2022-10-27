# TwinCar 필터모델

## 1. 데이터셋 및 학습 모델 정보

### 1.1. 데이터셋의 위치 및 구조

데이터셋은 `train`, `val`, `test`로 나누어져 있고 전체 구조는 아래와 같습니다.

```
root
  | - train
    - val
    - test
        | - 1normal
                | - image1.jpg
                  - image2.jpg
                  - ...
          - 2instrumental_panel
                | - image1-1.jpg
                  - image2-1.jpg
                  - ...
          - 3others
                | - image1-2.jpg
                  - image2-2.jpg
                  - ...

```

### 1.2. 학습 모델 위치

학습된 모델은 `./trained_model`폴더에 `filter-model-best.h5`라는 이름으로 저장되어 있습니다.

## 2. 추가 학습 및 평가

### 2.1. config 파일 구조

```yaml
sodaflow:
  dataset_path:
    - "/datasets/kaflix-filter-dataset/"
  start_ckpt_path: "./trained_model"
  test_model_path: "./export"

train_inputs:
  MODEL_NAME: "DenseNet169"
  INPUT_SIZE: [224, 224, 3]
  NUM_CLASSES: 3
  BATCH_SIZE: 16
  NUM_EPOCHS: 1
  LEARNING_RATE: 1e-5

MODEL_OUTPUT_DIR: "./export"
LOGS_DIR: "./logs"
EVAL_OUT: "./eval_out"
```

* sodaflow: trial실행시 지정되는 경로
  * dataset_path: 데이터셋의 root 경로
  * start_ckpt_path: fine-tuning시 불러올 이전 모델파일이 있는 directory 경로
  * test_model_path: 모델 평가시 불러올 모델파일이 있는 directory 경로
* train_inputs: 하이퍼파라미터
  * MODEL_NAME: 모델 구조 (DenseNet121, DenseNet169, MobileNetV2, ResNet50 가능)
  * INPUT_SIZE: 모델의 입력 사이즈
  * NUM_CLASSES: 분류할 클래스 수
  * BATCH_SIZE: 한 step에서 사용할 batch의 수
  * NUM_EPOCHS: 전체 데이터를 몇 번 학습할지
  * LEARNING_RATE: weight 업데이트시 사용할 학습률
* MODEL_OUTPUT_DIR: 학습된 모델이 저장되는 local 경로
* LOGS_DIR: 로그가 남겨지는 local 경로
* EVAL_OUT: 모델 평가 로그가 남겨지는 local 경로

### 2.2. 학습 실행

GPU 전체를 이용하여 학습하는 경우

```
python train.py
```

GPU 하나를 지정하여 학습하는 경우. GPU번호를 `CUDA_VISIBLE_DEVICES`에 지정하여 실행.

```
CUDA_VISIBLE_DEVICES=0 python train.py
```

학습이 완료되면 2.1의 설정파일에서 지정한 `MODEL_OUTPUT_DIR`아래에 `filter-epoch-01-0.270-0.980.h5`와 같은 이름으로 학습된 모델 파일이 저장됩니다.

### 2.3. 평가 실행

2.1의 설정파일에서 `sodaflow.test_model_path`의 경로를 학습이 완료된 모델 파일이 있는 directory 경로로 바꿔줍니다. 이후 아래의 명령을 실행하여 모델의 인식률을 테스트할 수 있습니다.

```
python evaluate.py # or
CUDA_VISIBLE_DEVICES=0 python evaluate.py # 0번 GPU만 사용
```

평가가 완료되면 2.1의 설정파일에서 `EVAL_OUT`에 지정된 directory아래에 결과 로그 JSON파일이 생성되며 `Twincar_Filter_model_CM.jpg`이미지로 confusion matrix를 확인할 수 있습니다.
