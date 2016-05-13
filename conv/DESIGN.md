## High Level Ideas

```python
"""
consider the convolution between layer n-1 and layer n:
*	chan_n:		number of feature map channels in layer n
*	chan_n1:	number of feature map channels in layer n-1
*	row/col_n:	feature map x,y dimensions in channel n (output map)
*	kern_x/y:	kernel x,y dimensions
"""
```

**Original convolution:**

```python
# for simplicity, consider stride of 1, ignore padding.
for PERMUTATION(chan_n, chan_n1, row_n, col_n, kern_x, kern_y):
  	output_fm[chan_n,row_n,col_n] += \
    		weight[chan_n,chan_n1,kern_x,kern_y] \
      	  * input_fm[chan_n1,(row_n+kern_x),(col_n+kern_y)]
```

**1st design:**

independent / irrelevant dimension: `chan_n`, `chan_n1`. 

```python
for PERMUTATION(row_n, col_n, kern_x, kern_y):
  	(weight,input_fm).map(on chan_n).reduce(on chan_n1)
```

Seems to be *not suitable* for Spark: each MapReduce is only doing `chan_n*chan_n1` computations.

**2nd design:**

replicate memory, to decouple `row_n`,`kern_x`,`col_n`,`kern_y`, then basically **all 6** dimensions are independent / irrelevant.

```python
"""
input_fm_expand is obtained by replicate and shift input_fm matrix:
i.e.:
	input_fm_expand[chan_n1,row_n,col_n,kern_x,kern_y] = \
    		input_fm[chan_n1,(row_n+kern_x),(col_n+kern_y)]
memory replication ratio:
	(kern_x * kern_y)
"""
(input_fm_expand, weight).map(on chan_n, chan_n1, row_n, col_n, kern_x, kern_y)\
						 .reduce(on kern_x, kern_y, chan_n1)
```

No loop is needed, but memory replication is too high.

*Modification:*

`input_fm` is expanded on `row_n` or `col_n`, but not both. 

Iterate on `row_n`, memory replicate on `col_n`. 

**3rd design**:

Parallelization based on *loop tiling*, not based on *loop unrolling* (i.e.: analysis of independent dimensions). 

*Serial version* of convolution:

```python
for col_n:
  	for row_n:
      	get_patch(input_fm)
        dot(patch,weight)
        update(output_fm)
```

*Tiling version*:

```python
for col_n stride by C:
  	for row_n stride by R:
      	get_multiple_patch(input_fm)	# (R*c) patches
        dot(patch_expanded, weight)
        update(output_fm)
```

Most intuitive: simply reduce the number of iterations. 













### current thought

- normal data set: best to parallelize `batch  x chan_n x chan_n1`
- blood cell: possibility exists that training is best when using very small batch size (or even online). --> parallelizing on `row_n` and `col_n` is best in this case. --> since `batch` is small, replicating data should also be fast. 
- I should try both.
- serial version: best with batch size of 10. --> cuz 3 types, 100 will undermine the "stochastic" nature of the training. 
- to parallelize:
  - starting from the same initial weight, let the 10 nodes each individually gradient descent on 750 images with 10 mini-batch, then combine the final net. --> way to combine: based on the delta w(i,j) after 75 batch updates: take the max delta among the 10 delta w values?
  - Or just keep the one with the highest validation accuracy?
    - Comparison of this and the serial version should be simple


- But the *max* seems to make more sense. 
  - Think about the feamap --> take max, do you need to re-arrange (re-order) ?
  - Is the order of feamap preserved? --> too bad if not.
  - THEN: Instead of combine in a pixel-wise fashion, we can do map by map, after coming up with a measure for relative change of weight (delta w) for a whole map.
  - [practical handbook of GA] -- pp.144


- need to have a sense of the percentage change (delta w)
- how does GPU achieve the speedup? --> should be parallelization as well --> but maybe GPU communication cost is much lower, like in the case of FPGA.
- GA: seems to be a completely different story. 
- *How to verify*
- *swap the members in the subgroup* (maybe after some point in the training)


- Get the roofline model for Spark
- check the delta for after 7500 images, compare it with that of first 750, second 750, ...
- How do you compare?





**Details:**

only parallelize when convolution starts. 

RDD is not good at `ndarray` operations (??).

is the `collect()` method too often (??)





## Implementation

### Profiling (Quantitative)

- Scheduler Delay: Big portion of the total runtime & caused by `collect()` operation
  - `T(num_workers, total_data_size)`

| instance type | num workers | total data size | scheduler delay |
| ------------- | ----------- | --------------- | --------------- |
|               |             |                 |                 |







### TODO: v2

effect of down-sampling, noise, distortion, ...

what cell is the net classifying well.

is the pre-knowledge of images helping with the architecture design? --> the kernel size, the initialization of weights

how to evaluate the training quality change (quantitatively)?