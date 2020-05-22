# Deep Deducing on Game of Life


This repository contains codes illustrating how deep deducing plays Game of Life.
To reproduce the results in the paper, simply run Deducing_xxx.py



## Requirements

Packages requirements:

```
numpy
```


```
scipy
```

## Learning phase content

To train sets of weight matrix in the paper, run this command:

```
Learning.py            
```


## Deducing phase content
To use sets of trained weight matrix to have each cell in the present game table to deduce its legit status in the next generation, run this command:

```
Deducing_xxx.py              
```




## Already-trained sets of weight matrix

We have sets of weight matrix already-trained for your convenience. Please see the definition below to understand the content of each set of weight matrix.


self.```Conway``` _ ```1``` _ ```100x100x100``` _ ```30``` _ ```0.000001``` _ ```50m``` _ ```[1]``` _ ```weight_list```


means a single set of weight matrix.


The definition for each bracket (from left to right) is listed below:

```Conway```    
means this neural network is trained to mimic the rationale of the cells in Conway's Game of Life.

```1```     
means the sample batch in the learning phase for this trained neural network is 1 per each learning epoch.
          
```100x100x100```  
means the trained neural network has three hidden layers, each with 100 neurons.
          
```30```          
means the initial value for the set of slope multiplie to be updated in the learning phase.
          
```0.000001```    
means the learning rate in the learning phase for this trained neural network is 0.000001.

```50m```        
means the learning epochs in the learning phase for this trained neural network is 50 million or 5*10^7. The learning epochs are usually big in order to force the neural network to over-fit.

```[1]```    
means the label of this trained neural network under the above training condition.
          For example, if it is [3], then it means this neural network is the third neural network under the training condition 
          ```Conway``` _ ```1``` _ ```100x100x100``` _ ```30``` _ ```0.000001``` _ ```200m```.
          
```weight_list```  
means the set of weight matrix of this trained neural network.





## Results

For results, please refer to the figures in the paper.



