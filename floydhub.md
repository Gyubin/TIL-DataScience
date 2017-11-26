# FloydHub 사용법 간단 정리

한 마디로 "Heroku for Deep Learning"이다. 서버 설치, 설정, 그런거 다 필요없고 난 코드와 데이터셋만 준비하고 floydhub으로 돌리면 끝난다. 아래 명령어로 CLI 도구부터 설치하자.

```sh
pip3 install -U floyd-cli # 버전 3의 가상환경이라면 그냥 pip를 써도 무방
```

## 0. Warning

FloydHub는 시간단위 과금되는 모델이다. `.py` 파일을 만들어서 실행하면 실행된 시간만큼이 차감된다. 그런데 **jupyter notebook**을 실행하면 연산 시간이 아니라 jupyter를 실행한 시간이 그대로 차감되기 때문에 FloydHub에서는 웬만하면 쓰지 않도록 한다. `.py` 파일을 만들어서 실행하는 걸 강력히 추천.

## 1. 온라인 저장소 만들기

- FloydHub에선 프로젝트 코드와, 데이터를 분리해서 저장한다.
- FloydHub에서 코드를 실행하면 그 때의 코드들을 전부 카피해서 업로드하는데 만약 디렉토리 내에 데이터가 있으면 데이터 역시 카피해서 업로드하게 된다.
- 거의 변하지 않고, 대용량인 데이터를 매번 업로드하게되면 FloydHub에게도 손해고, 시간단위로 사용하는 유저에게도 손해다. 그래서 데이터 저장소를 따로 파놓는다.

### 1.1 Dataset

- FloydHub에 접속하고 우측 상단에 `+` 모양의 버튼이 있다. **New dataset**을 선택하면 데이터셋 이름과 설명을 적을 수 있다. 완료하면 저장소가 하나 생긴다.
- 이후엔 Local 디렉토리에 내가 원하는 데이터셋을 저장한다. 그리고 아래 코드를 순서대로 입력한다.
    + `cd my/data/path` : 데이터셋이 저장된 디렉토리로 진입하고
    + `floyd data init MY_DATSET_NAME` : 뒷 부분 'MY_DATSET_NAME' 부분을 내 데이터셋 저장소 이름으로 대체해서 입력한다.
    + `floyd data upload` : 데이터 업로드 시작. 압축해서 업로드하기 때문에 용량이 크면 시간이 꽤 걸린다.

### 1.2 Project

- 데이터셋 준비에서와 마찬가지로 우측 상단 `+` 버튼을 누르고 **New project** 선택하고 원하는 이름 설정해서 생성한다.
- Local에서 작업
    + `cd my/project/path` : 프로젝트 디렉토리로 진입한다.
    + `floyd init MY_PROJECT_NAME` : "MY_PROJECT_NAME" 부분을 위에서 만든 프로젝트 저장소이름으로 대체해서 입력
- 초기화만 했고 해당 프로젝트 실행은 아래에서 다시 설명하겠다.

## 2. Dependencies 설정

- 기본적인 numpy 같은 건 설치되어있지만 **gensim** 같은건 설치하라고 지정해줘야한다.
- local project 디렉토리에 **floyd_requirements.txt** 파일을 만든다.
- `pip.freeze` 명령어를 쳐서 필요한 패키지를 버전과 함께 복사하고, pip에서 사용하는 requirements.txt처럼 동일하게 적어주면 된다.

```
gensim==3.1.0
```

## 3. 코드 실행하기

기본형: `floyd run [OPTIONS] [COMMAND]`

### 3.1 OPTIONS

- Instance Type
    + `--cpu` : DEFAULT. Preemptible server
    + `--gpu` : Preemptible server
    + `--cpu+` : Dedicated server
    + `--gpu+` : Dedicated server
- Dataset: `--data <name_of_datasource>:<mount_point_on_server>`
    + 앞부분 : 미리 준비한 데이터셋의 주소, 즉 데이터가 저장된 서버 지칭.
    + 뒷부분 : 코드가 실행되는 서버에서 어떤 경로에 마운트 될 것인지 지정. 무조건 root directory에 마운트되어야 하고, nested directory는 기본적으로 지원하지 않으니 하나만 쓰자.
    + `--data` 부분을 연달아 써서 여러 데이터를 마운트시킬 수도 있다.
    + 예제 : `gyubin/datasets/udacity-gan/1:/my_data`
- Mode: `--mode <mode_name>`
    + `--mode job` : DEFAULT. 일반 코드 실행.
    + `--mode jupyter` : 주피터 노트북 실행(연산이 아니라 실행되는 시간만큼 차감.)
    + `--mode serve` : API 서버로 사용하는 것이고 flask 기반으로 동작한다. 그래서 requirements.txt에 flask를 적어줘야함.
- Environment: `--env <environment_name>`
    + 자주 사용하는 환경들(keras, tensorflw, pytorch 등)을 바로 제공해준다. 프로젝트를 실행했을 때 설치하는 시간을 아껴줌
    + `floyd run --env tensorflow-1.3 "python train.py"`
    + `floyd run --env pytorch-0.2 "python train.py"`
    + 자세한 정보: https://docs.floydhub.com/guides/environments/
- Message: `--message` or `-m`
    + git commit 하는 것처럼 적으면 되
    + 실행할 때마다 생기는 새로운 버전들에 설명을 달아준다. 웹사이트에서 보기 편하다.
- Tensorboard: `--tensorboard`
    + 옵션을 지정해주면 텐서보드가 실행되는 주소가 할당되어 쉽게 접근해서 확인할 수 있다.
    + 자세한 정보: https://docs.floydhub.com/guides/jobs/tensorboard/

### 3.2 COMMAND

- **job** 모드로 실행할 때 커맨드라인에 어떤 명령어를 입력할건지 문자열로 지정해주는 부분이다.
- 내 시작 파일을 어떤 명령어로 실행할지, 어떤 옵션을 줄지 다 문자열로 주면 끝

## 4. 프로젝트 관리하기

### 4.1 실행 상태 중지하기

- 웹사이트 들어가서 cancel 버튼 눌러주면 된다.
- `floyd stop gyubin/projects/PRJ_NAME/2` : job 지정해서(지금 2를 지정한 것) stop한다.
- `floyd stop foo` : foo 프로젝트의 가장 최근 job 정지
- `floyd stop foo/2` : foo 프로젝트의 2번 job 정지

### 4.2 실행 상태 보기

- `floyd status` : 내 모든 프로젝트의 실행 상태를 본다.
- `floyd status PRJ_NAME` : 지정한 프로젝트의 최신 job 상태를 보여준다.

### 4.3 특정 job 재시작하기

- `floyd restart gyubin/projects/PRJ_NAME/1 "python train.py"` : 기본
- `floyd restart gyubin/projects/PRJ_NAME/1 "python train.py" --gpu` : 인스턴스 바꿔서 재시작 가능
- `floyd restart gyubin/projects/PRJ_NAME/1 "python train.py" --data server:mnist` : 데이터셋 바꿔서 재시작 가능
- COMMAND 부분에서 parameter를 바꾸는 형태로 재시작하는게 가장 자주 쓰는 유즈케이스다.

## 5. Output 저장하기

- 서버에서 코드가 실행되고, 어떤 저장해야할 파일이 생겼을 때(대표적으로 chekpoints) `/ouputs` 디렉토리에 저장해야 유저가 접근할 수 있다.
- `/output`에 file을 쓰거나, model의 save 메소드를 사용하거나 하면 된다.
- output 파일만 단독으로 삭제할 수 없고, job 자체를 삭제해야한다.
- 다운로드
    + 웹페이지에서 다운로드 버튼을 누르거나
    + `floyd data clone USER_NAME/projects/PRJ_NAME/1/output` : 형태로 job 번호 지정해서 output 받을 수 있다.
