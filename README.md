# Problem understanding:
 </br>
In this project, we test our understanding of the works of optimizers in deep learning. We do so by exploring various optimizers and the mathematics behind them. We try to find the key differences between Gradient descent and its invariances and Adaptive learning optimizers. We then implement a linear regression model alongside 5 adaptive learning optimizers from scratch. We evaluate each by plotting the convergence of all of the optimizers.


<!-- <br> -->


# Running the project #
Here description of the project.

If you do not have venv package, please refer to this [link](https://linuxize.com/post/how-to-create-python-virtual-environments-on-ubuntu-18-04/)
</br>

## Create virtual environment ##

```
$ python3 -m venv ENV_NAME
```
## Activate your environment ##

```
$ source ENV_NAME/bin/activate
```

## Requirement installations ##
To run this, make sure to install all the requirements by:

```
$ pip install -r requirements.txt 
```
# Training the model with a specific optimizer #

```
$ python3 main.py --optimizer_name --num_epochs
```
## Example of running models ##

```
$ python3 main.py --optimizer_name adam --num_epochs 1000
```

```
$ python3 main.py --optimizer_name all --num_epochs 1000
```

# Results Presentation

adam result      |  Result from All Optimizers
:--------|--------
![adam result](figures/Adam.png)|![Plot of all Optimizers](figures/All_plots.png) 

<!-- ``` adam result```  </br>
![adam result](figures/Adam.png) 

```Plot of all Optimizers```  </br>
![Plot of all Optimizers](figures/All_plots.png)  -->
---


# Related Papers #

* <a href= 'https://arxiv.org/pdf/1412.6980.pdf'> Adam </a>
* <a href= 'https://arxiv.org/pdf/1212.5701.pdf'> Adadelta</a>
* <a href= 'https://arxiv.org/pdf/1212.5701.pdf'> Momentum</a>
* <a href= 'https://arxiv.org/pdf/1212.5701.pdf'> RMSProp</a>
* <a href= 'https://arxiv.org/pdf/1212.5701.pdf'> Adagrad</a>



# Contributors #
<div style="display:flex;align-items:center">

<div style="display:flex;align-items:center">
  <div>
    <h5> <a href='.'> Ms. Khadija Iddrisu </a> </h5> <img src="figures/K.PNG" height= 7% width= 7%>
   
  <div>
        <h5> <a href='..'> Mr. Paul sanyang</a> </h5> <img src="figures/Paul.jpeg" height= 7% width= 7%>
  <div>
    <h5> <a href='.'> Mr. Albert Agisha N </a> </h5> <img src="figures/Albert.jpeg" height= 7% width= 7%>
    
  <div>
    <h5> <a href='.'> Mr. Idriss Nguepi N  </a> </h5> <img src="figures/Idriss_picture.JPG" height= 7% width= 7%>
    
</div>