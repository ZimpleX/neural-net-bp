Preliminary measurement: 

- *t2.large 4 slaves*
- total of 8 cores
- base matrix: `200*3*128*128`
- kernel matrix: `5*3*11*11`
- output matrix: `200*5*138*138`

| version     | get patch time (s) | dot time (s) | total time (s) |
| ----------- | ------------------ | ------------ | -------------- |
| serial      | 4.5                | 5.3          | 10             |
| 8 partition |                    |              | 7              |



- now change batch size to `800`, and other conf unchanged

| version      | get patch time(s) | dot time(s) | total time(s) |
| ------------ | ----------------- | ----------- | ------------- |
| serial       | 19                | 20          | 40            |
| 8 partitions |                   |             |               |







### CPUload, Duration, Partition

`Duration = T(CPUload, Partition)`

- `Duration = K(Partition)*CPUload+B(Partition)`


- `K(Partition) = c*(1/Partition)`