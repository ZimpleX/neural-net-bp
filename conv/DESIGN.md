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