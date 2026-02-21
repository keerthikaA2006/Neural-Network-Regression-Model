# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
The class NeuralNet inherits from nn.Module, which is the base class for all neural networks in PyTorch.

In the constructor init, layers and activation functions are defined.

The first layer self.n1=nn.Linear(1,10) takes one input feature and maps it to 10 neurons.

The second layer self.n2=nn.Linear(10,20) processes the 12 outputs and maps them to 20 neurons.

The third layer self.n3=nn.Linear(20,1) reduces the 14 features back to a single output.

The activation function self.relu=nn.ReLU() introduces non-linearity, helping the network learn complex patterns.

A history dictionary is initialized to store the loss values during training for performance tracking.

The forward function defines how input data flows through the network layers.

Input x is first passed through n1 and activated by ReLU, then through n2 with ReLU again.

Finally, the processed data passes through n3 to produce the output, which is returned.
## Neural Network Model

<img width="763" height="556" alt="image" src="https://github.com/user-attachments/assets/82a5deee-a61c-4656-979a-daf6aefe3c28" />

## DESIGN STEPS
### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Keerthika A
### Register Number: 212224220048
```python
class Neuralnet(nn.Module):
   def __init__(self):
        super().__init__()
        self.n1=nn.Linear(1,10)
        self.n2=nn.Linear(10,20)
        self.n3=nn.Linear(20,1)
        self.relu=nn.ReLU()
        self.history={'loss': []}
   def forward(self,x):
        x=self.relu(self.n1(x))
        x=self.relu(self.n2(x))
        x=self.n3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
sai_brain=Neuralnet()
criteria=nn.MSELoss()
optimizer=optim.RMSprop(sai_brain.parameters(),lr=0.001)

def train_model(sai_brain,x_train,y_train,criteria,optmizer,epochs=4000):
    for i in range(epochs):
        optimizer.zero_grad()
        loss=criteria(sai_brain(x_train),y_train)
        loss.backward()
        optimizer.step()
        
        sai_brain.history['loss'].append(loss.item())
        if i%200==0:
            print(f"Epoch [{i}/epochs], loss: {loss.item():.6f}")



```
## Dataset Information
<img width="352" height="394" alt="Screenshot 2026-02-10 153210" src="https://github.com/user-attachments/assets/c00b9bbc-6c31-480f-98ce-1d494b314741" />


## OUTPUT
<img width="790" height="488" alt="Screenshot 2026-02-21 205145" src="https://github.com/user-attachments/assets/e2589078-e57b-4a02-9ec3-604686eecab0" />


### Training Loss Vs Iteration Plot
<<img width="757" height="620" alt="Screenshot 2026-02-21 205238" src="https://github.com/user-attachments/assets/cf31bf13-b14b-49a1-a908-a7dbedb6ee12" />


### New Sample Data Prediction
<img width="901" height="143" alt="Screenshot 2026-02-21 205329" src="https://github.com/user-attachments/assets/6b94b267-64c8-452d-b406-731d2bbe65c7" />


## RESULT
Successfully executed the cod to develop a neural network regression model.

