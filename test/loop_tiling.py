"""
test loop tiling
"""
from logf.printf import printf

def tiling(m, n, s):
    ret = []
    for i in range(0, m, s):
        for j in range(0, n, s):
            for ii in range(i, min(i+s, m)):
                for jj in range(j, min(j+s, n)):
                    ret += [(ii,jj)]
    return ret

def non_tiling(m, n):
    ret = []
    for i in range(m):
        for j in range(n):
            ret += [(i,j)]
    return ret



if __name__ == '__main__':
    m, n = (256,256)
    s = 2
    printf('test non tiling version')
    baseline = non_tiling(m, n)
    len1 = len(baseline)
    printf('num of tuples: {}', len1, separator=None)
    baseline = set(baseline)
    assert len(baseline) == len1

    printf('test tiling version')
    testline = tiling(m,n,s)
    len2 = len(testline)
    printf('num of tuples: {}', len2, separator=None)
    testline = set(testline)
    assert len(testline) == len2
    
    if baseline == testline:
        printf('TEST PASSED!', type='WARN')
