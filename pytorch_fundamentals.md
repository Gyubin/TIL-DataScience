# PyTorch fundamentals

설치는 http://pytorch.org/ 에 들어가서 자기 사양 선택하면 설치 커맨드가 나온다. 내 맥 환경의 경우 다음과 같다. 

```sh
pip3 install http://download.pytorch.org/whl/torch-0.2.0.post3-cp36-cp36m-macosx_10_7_x86_64.whl 
pip3 install torchvision 
```

## 1. 기본

- 변수 선언
    + `torch.Tensor(i, j, k)` : shape이 (i, j, k)인 데이터를 생성한다. rank는 자유롭게 설정가능
    + `torch.rand(i, j)` : shape이 (i, j)인 0에서 1 사이의 난수 데이터 생성
- add 연산 종류
    + `a + b` : 단순 덧셈 연산으로 element-wise add 연산 가능
    + `torch.add(a, b, out=result)` : 리턴값도 있고, 덧셈 연산 결과를 특정 변수에 저장 가능
    + `a.add_(b)` : `add_` 함수 호출. 호출한 변수의 값 자체가 변한다.
- 인덱싱은 numpy와 같은 방식으로 사용한다. ex) `a[:, 1:4]` , `a[3, 4]`
- `torch.ones(i, j)` , `torch.zeros(i, j)` : (i, j) shape으로 데이터 만드는데 1과 0으로 초기화
- torch to numpy: 모든 torch.tensor에서 `.numpy()` 함수를 호출하면 ndarray로 변환된 값이 리턴된다. 원본 torch tensor의 영향을 받는다.
    + `x1=torch.ones(2, 3) ; x2=x1.numpy()`
    + `x1.add_(1)`
    + `print(x1, x2)` : x1에만 값을 1 더했는데 x2 값도 변했음을 알 수 있다.
- numpy to torch: 위 경우와 반대.
    + `a=np.ones(5) ; b=torch.from_numpy(a)`
    + `np.add(a, 1, out=a)`
    + `print(a, b)` : 역시 b도 영향을 받는다.
- `.norm()` : norm 값 구할 수 있다.
- tensor를 GPU로 옮기고싶다면 `.cuda()`

    ```py
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        x + y
    ```

## 2. Variable

```py
import torch
from torch.autograd import Variable

x = Variable(torch.Tensor([3, 4, 5, 6]), requires_grad=True)
w = Variable(torch.Tensor([10]), requires_grad=True)
z = w*x + 2
out = z.mean()

out.backward()
print(x.grad, w.grad)
```

- parameters : function을 실행할 때 넣는 값
    + `data` : Tensor 데이터
    + `requires_grad` : grad를 계산할건지 말건지를 나타내는 bool 값
- variables : `.`으로 호출 가능한 값들
    + `data` : 갖고 있는 데이터 값
    + `grad` : data와 매칭되는 미분값. reassign 안된다.
    + `requires_grad` : grad를 계산한다고 한 값인지 Boolean으로 리턴. leaf node에서만 바뀔 수 있다.
    + `is_leaf` – leaf인지, 즉 user가 1차로 만든 값인지
    + `grad_fn` – 만들어진 미분 공식
- Variable을 이용하면 미분을 쉽게 구할 수 있다.
    + `out.backward(gradient=None)` : gradient 파라미터의 디폴트값이 None이다. 그래서 파라미터로 넘겨주는게 없으면 out은 무조건 스칼라값이어야한다.
    + `out.backward(gradient=torch.randn(1, 10))` : 만약 classification 문제에서 클래스가 10개라면 저런식으로 차원을 지정해줘야한다.
- 기준이 되는 스칼라값에서 `backward()` 함수를 호출한다.
- 호출한 이후에 `x`와 `w`에서 미분값을 가져올 수 있다.
    + `x.grad` = d(out) / d(x)
    + `w.grad` = d(out) / d(w)
    + `z` 같은 경우, 즉 user가 1차로 만든 값이 아니면 미분값을 구하지 않는다.

## 3. nn

- `torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)`
    + `in_channels`, `out_channels`: input, output 지정
    + `kernel_size` : int 하나만 주면 정방, tuple로 주면 첫 번째가 height, 두 번째가 width
    + `stride`, `padding`: 역시 tuple로 다른 값을 줄 수 있다.
    + `dilation` : [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) 참조
    + variable로 `weight`, `bias` 호출 가능
- `torch.nn.Linear(in_features, out_features, bias=True)`
    + 레이어의 dims를 Fully connected layer 방식으로 변환하는 함수다.
    + 즉 이 함수로 만들어지는 값은 `(input_c, output_c)`의 **WEIGHT MATRIX**다.
    + 데이터를 나타내는 것은 row이고, feature가 column인데 이 feature channel을 바꿔주는 것. row는 건드리지 않는다. 다만 size는 매칭되어야한다.

    ```py
    input = torch.autograd.Variable(torch.randn(128, 20))
    m = torch.nn.Linear(20, 30)
    output = m(input)
    print(output.size()) # 128, 30
    ```

## 4. Basic CNN

### 4.1 Forward

![convnet](http://pytorch.org/tutorials/_images/mnist.png)

```py
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
```

- 구조
    + `nn.Module` 클래스를 상속
    + `__init__` : 부모 클래스의 것을 기본적으로 가져오고, 내가 필요한 레이어들을 정의한다. 위 예제에서는 conv 2개, fc 3개다.
    + `forward` : 데이터를 받아서 내가 설정한 아키텍처 구조에 따라서 연산을 진행하는 과정이다.
- `num_flat_features` : batch 수를 나타내는 첫 번째 dimension은 제외하고 뒤의 channel, height, width를 다 곱한 값을 리턴해주는 함
- `x.view(d1, d2)` : reshape 역할을 하는 함수다.

### 4.2 Backward

```py
net = Net()
input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)

net.zero_grad()
out.backward(torch.randn(1, 10))

target = Variable(torch.arange(1, 11))
criterion = nn.MSELoss()

loss = criterion(output, target)

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
```

- 모델 인스턴스를 만들고, 인풋 데이터를 1개, 1채널 32x32 사이즈로 준비한다. 그리고 데이터를 모델에 넣어서 out 값을 저장한다.
- `net_instance.zero_grad()` : 모든 weight의 gradient buffer를 0으로 만든다.
- `out.backward(torch.randn(1, 10))` : 분류 클래스가 10개라서 out의 차원이 1x10이다. 맞춰줘야한다.
- `target` : 그냥 더미로 대충 설정해서 만든다. ground-truth다.
- `nn.MSELoss()` : 로스는 MSE로 정한다.
- `grad_fn` 을 통해 미분을 계산해야하는 이전 함수들을 확인할 수 있다.

```py
print('before: ', net.conv1.bias.grad)
loss.backward()
print('after: ', net.conv1.bias.grad)

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
```

- loss에 대해 backward propagation을 하면 각 params에 대해 미분값이 구해진다.
- 모든 params에 대해 gradient descent를 한 번 실행한다. 위 예제는 1 epoch다.

```py
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.01)

epoch = 100
for i in range(epoch):
    optimizer.zero_grad()
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

- 라이브러리를 활용해 더 쉽게 optimizer를 사용할 수 있다.
- 매번 `zero_grad`를 하는 이유는 저걸 해주지 않으면 그라디언트가 매번 중첩되기 때문이다. 유저가 하도록 만든 이유는 그라디언트를 중첩해서 한 번에 업그레이드하는 경우도 있기 때문.
- 처음 optiizer를 선언할 때 우리 모델의 parameters를 넣어주면 이 값을 업데이트하게 된다.
- `optim`에 다른 optimizer가 구현되어있으니 원하는대로 사용하면 된다.

## 5. Using prtrained model: ResNet

source: [pytorch tutorial by @yunjey](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py)

```py
resnet = torchvision.models.resnet18(pretrained=True)

for param in resnet.parameters():
    param.requires_grad = False

resnet.fc = nn.Linear(resnet.fc.in_features, 100)

images = Variable(torch.randn(10, 3, 224, 224))
outputs = resnet(images)
print (outputs.size())
```

- ResNet의 학습된 weight를 가져와서 사용해본다.
- 마지막 FC layer의 dims만 나에게 맞는 클래스 수로 바꾸고(1000 -> 100), 그 바꾼 FC만 학습하기 위해 위처럼 for 반복을 돌아서 grad 계산 여부를 모두 False로 바꾼다.
- 위처럼 output이 내가 정한 클래스 수로 나오는지 확인해보고, backward update를 하며 학습한다.

## 6. Save model

```py
# Save and load the entire model.
torch.save(resnet, 'model.pkl')
model = torch.load('model.pkl')

# Save and load only the model parameters(recommended).
torch.save(resnet.state_dict(), 'params.pkl')
resnet.load_state_dict(torch.load('params.pkl'))
```

- 위처럼 `torch.save(var, file_path)` , `torch.load(file_path)` 형태로 하면 된다.
- 전체 모델보다는 parameter만 save하는것이 추천됨. model instance에서 `model.state_dict()`를 호출하면 orderedDict 객체에 parameter가 담겨서 리턴된다.

## 7. Data loader

### 7.1 Sample Data

MNIST, Fashion-MNIST, COCO 등 유명한 데이터셋을 굉장히 쉽게 다운받을 수 있다.

```py
import torchvision.datasets as dsets
import torchvision.transforms as transforms

train_set = dsets.CIFAR10(root='./data/',
                          train=True, 
                          transform=transforms.ToTensor(),
                          download=True)

image, label = train_dataset[0]
```

- dsets에서 다양한 데이터를 불러올 수 있다.: `MNIST`, `FashionMNIST`, `CocoCaptions`, `CIFAR10` 등
- parameters(데이터셋마다 약간의 차이는 존재)
    + `root` : dataset path 지정
    + `download` : True 값을 주면 해당 path에 다운로드도 하게 된다.
    + `train` : True면 train set이고, False면 test set이다.
    + `transform` : 함수를 넣어서 데이터를 어떻게 변형할지 지정. `transforms.ToTensor()`는 PIL이나 ndarray를 torch Tensor로 바꾸는 함수
    + `target_transform` : target(y값)을 변화시킬 때 사용하는 함수 지정
- 원소는 `(X, y)` 튜플 형태로 구성되어있다.

### 7.2 Data loader

Raw dataset만 있으면 셔플, 미니배치, 스레드 등의 기능을 쉽게 사용할 수 있도록 해준다.

```py
import torch

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True,
                                           num_workers=2)

# 1. iter 함수로 하나하나 순서대로 뽑을 수도 있고
data_iter = iter(train_loader)
images, labels = data_iter.next()

# 2. 바로 for 반복 돌아도 된다.
for images, labels in train_loader:
    pass
```

- `torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False, timeout=0)`
    + `dataset` : X, y 데이터셋
    + `batch_size` : SGD에서 mini-batch 크기
    + `shuffle` : 섞을건지 Boolean으로
    + `sampler` : 데이터를 뽑는 분포, 규칙을 정하는 것을 `Sampler` 라는 torch의 클래스로 정해서 넣어준다. 이거 하려면 shuffle은 무조건 False로
    + `batch_sampler` : sampler와 비슷하지만 리턴이 batch다.
    + `num_workers` : process 수 지정
    + `drop_last` : 배치 사이즈가 정확하게 안 나눠지면 마지막 배치는 버릴지 결정
    + `timeout` : 항상 음수가 아닌 값을 줘야하고 양수값이면 일정 시간 이상 못 불러오면 취소한다.
- iterable에서 한 번 뽑아올 때 batch size만큼의 샘플들이 뽑혀나온다.

### 7.3 Custom dataset

```py
import torch.utils.data as data

class CustomDataset(data.Dataset):
    def __init__(self):
        # file path, name 등의 것들 정의
        pass
    def __getitem__(self, index):
        # 파일 읽고, 전처리하고, 최종 데이터셋을 리턴
        pass
    def __len__(self):
        # 데이터셋 사이즈 지정
        return 0 

# Then, you can just use prebuilt torch's data loader. 
my_d = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=my_d,
                                           batch_size=100, 
                                           shuffle=True,
                                           num_workers=2)
```

내 고유의 데이터를 쓰는데, `DataLoader`를 이용하고싶다면 위처럼 데이터셋을 생성하면 된다. 상속만 잘 받고 주요 함수 구현해서 쓰자.

## 8. Regression

### 8.1 Linear regression

```py
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  
    
    def forward(self, x):
        out = self.linear(x)
        return out
```

- `nn.Module` 상속받고, `__init__`에서 `super` 함수 호출해준다.
- `nn.Linear` 함수는 weight matrix를 나타내는 것이므로 linear regression에서 feature를 몇 개 쓸건지, 결과값은 몇 개를 낼건지만 정해서 i, o size로 지정
- `forward` 함수를 만들어 데이터 입력 함수 정한다. 나중에 이 함수 호출해서 optimize 한다.

### 8.2 Logistic regression
