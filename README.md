# BART 챗봇 학습
## 설명
본 프로젝트는 챗로그 데이터와 일상 대화 데이터를 바탕으로 BART 챗봇 모델을 학습한 프로젝트입니다.
그리고 일상대화 데이터를 사용한만큼 반말 데이터가 대부분이라서 학습 후 반말로 답을 하는 것을 방지하기 위해 LSTM 기반 형태소 분석기를 이용하여 반말의 답변인 경우 존댓말로 바꾸어주는 후처리 알고리즘까지 적용했던 프로젝트입니다.
다만 데이터, 후처리 코드 및 형태소 분석기 모델은 공개하지 않겠습니다.
본 코드의 자세한 설명은 [BART를 이용한 한국어 챗봇 모델 학습](https://ljm565.github.io/contents/bart2.html)을 참고하시기 바랍니다.
<br><br><br>



## 모델 종류
* ### Bidirectional Autoregressive Transformers (BART)
    챗로그 및 일상 대화 데이터를 이용하여 학습합니다.
<br><br><br>


## 사용 데이터
* 챗로그 데이터
* 일상대화 데이터
<br><br>



## 사용 방법
* ### 학습 방법
    학습을 시작하기 위한 argument는 4가지가 있습니다.<br>
    * [-d --device] {cpu, gpu}, **필수**: 학습을 cpu, gpu로 할건지 정하는 인자입니다.
    * [-m --mode] {train, inference, chatting}, **필수**: 학습을 시작하려면 train, 학습된 모델을 가지고 있어서 test set의 BLEU 등의 학습 결과를 보려면 inference, 학습된 모델을 통해 chatting을 해보고자 한다면 chatting을 선택해야합니다. inference, chatting 모드를 사용할 경우, [-n, --name] 인자가 **필수**입니다.
    * [-c --cont] {1}, **선택**: 학습이 중간에 종료가 된 경우 다시 저장된 모델의 체크포인트 부분부터 학습을 시작할 수 있습니다. 이 인자를 사용할 경우 -m train 이어야 합니다. 
    * [-n --name] {name}, **선택**: 이 인자는 -c 1 혹은 -m {inference, chatting} 경우 사용합니다.
    중간에 다시 불러서 학습을 할 경우 모델의 이름을 입력하고, inference, chatting을 할 경우에도 실험할 모델의 이름을 입력해주어야 합니다(최초 학습시 src/config.json에서 정한 모델의 이름의 폴더가 형성되고 그 폴더 내부에 모델 및 모델 파라미터가 json 파일로 형성 됩니다).<br><br>

    터미널 명령어 예시<br>
    * 최초 학습 시
        ```
        python3 src/main.py -d cpu -m train
        ```
    * 중간에 중단 된 모델 이어서 학습 시
        <br>주의사항: config.json을 수정해야하는 일이 발생 한다면 base_path/src/config.json이 아닌, base_path/src/model/{model_name}/{model_name}.json 파일을 수정해야 합니다.
        ```
        python3 src/main.py -d gpu -m train -c 1 -n {model_name}
        ```
    * 최종 학습 된 모델의 test set에 대한 결과를 확인할 시
        <br>주의사항: config.json을 수정해야하는 일이 발생 한다면 base_path/src/config.json이 아닌, base_path/src/model/{model_name}/{model_name}.json 파일을 수정해야 수정사항이 반영됩니다.
        ```
        python3 src/main.py -d cpu -m inference -n {model_name}
        ```
    <br><br>

* ### 모델 학습 조건 설정 (config.json)
    **주의사항: 최초 학습 시 config.json이 사용되며, 이미 한 번 학습을 한 모델에 대하여 parameter를 바꾸고싶다면 base_path/src/model/{model_name}/{model_name}.json 파일을 수정해야 합니다.**
    * pretrained: {0, 1} 중 선택. 1을 선택하면 pre-trained KoBART로 학습.
    * base_path: 학습 관련 파일이 저장될 위치.
    * model_name: 학습 모델이 저장될 파일 이름 설정. 모델은 base_path/src/model/{model_name}/{model_name}.pt 로 저장.
    * loss_data_name: 학습 시 발생한 loss data를 저장하기 위한 이름 설정. base_path/src/loss/{loss_data_name}.pkl 파일로 저장. 내부에 중단된 학습을 다시 시작할 때, 학습 과정에 발생한 loss 데이터를 그릴 때 등 필요한 데이터를 dictionary 형태로 저장.
    * q_max_len: 질의의 최대 길이.
    * a_max_len: 응답의 최대 길이.
    * batch_size: batch size 지정.
    * epochs: 학습 epoch 설정.
    * lr: learning rate 지정.
    * greedy: {0, 1} 중 선택. Greedy search로 답변을 생성하려면 1, 아니면 0으로 설정.
    * temperature: 1로 설정 시 모델이 내어준 확률을 그대로 이용. 1보다 작을 시 확률의 차이를 좀 더 크게 만들어 덜 다양한 답변을 제한함. 1보다 클 시 확률의 차이를 거의 없게 만들어 다양한 답변을 내어주게 함.
    * topk: 생성시 top-k.
    * topp: 생성시 top-p.
    * early_stop_criterion: Validation set의 최대 accuracy를 내어준 학습 epoch 대비, 설정된 숫자만큼 epoch이 지나도 나아지지 않을 경우 학습 조기 종료.
    * result_num: 학습 시 보여줄 샘플 개수.
    <br><br><br>

## 결과
* ### BART 챗봇 학습 결과 
    존댓말 변환기를 달지 않은 결과입니다.
    그리고 같은 질문이라도 temperature, top-k, top-p 파라미터에 의해 다른 답변을 내어주는 것을 확인할 수 있습니다.
    ```
    Q: 유재석 요즘 너무 재밌지 않아?
    A: 유재석은 너무 웃기고 재밌지.

    Q: 요즘 너무 피곤해서 구론산 쟁여놓고 먹고 있어
    A: 응응 구론산 먹으면 몸이 좀 가벼워진다더라!

    Q: 요즘 너무 피곤해서 구론산 쟁여놓고 먹고 있어
    A: 응 구론산 먹으면 몸 좋아진다던데 나도 사야겠어
    ```


<br><br><br>
