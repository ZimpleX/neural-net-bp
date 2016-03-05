# Spark -- checklist

### Reference:

[best practice](http://robertovitillo.com/2015/06/30/spark-best-practices/)

### Transformation & Actions

- `flatMap` vs. `map`:
  - Both return a new RDD
  - `flatMap`: single element can be mapped to a list of output
  - `map`: single element is mapped to *key-value* pair only
  - **e.g.:** `textRDD.flatMap(lambda _: _.split())` will return a **single** list even if `textRDD` is iterating over each line. While `textRDD.map(lambda: _: (_, f(_)))` will return a `RDD` with the same number of elements as the number of lines in `textRDD`

### RDD operations

- `DAG`: basically the data dependency
- `stage`: if an operation requires data redistribution across workers, then this is a **new** `stage`.
  - **e.g.:** `shuffle` or `groupByKey`
- `tasks:` parallel operations within a `stage` are merged into one `task`. 
- "In general, 2-3 tasks per CPU core in the cluster is recommended." / "Rule of thumb, tasks should take at least 100ms to execute. "



### TODO:

- Set up `unittest` !!!!


- read data from `hdfs` instead of `file:///`
- parallelize on `batch(x chan_n x chan_n1)` dimension
- ​