# Author: Caihua Wang <490419716@qq.com>

from math import sqrt
from numpy import sign, dot
from numpy.linalg import norm


__all__ = ['sparseness']

def sparseness(v, lam):
    assert 0 < lam <= 1
    lam += sqrt(len(v)) * (1 - lam)
   
    if norm(v,1)/norm(v,2) <= lam:
        return v 
    else:
        v_abs = abs(v); v_abs.sort()
        v_abs=v_abs[::-1]
        ro = int(lam**2); i = ro+1
        s1 = v_abs[0:ro].sum()
        s2 = (v_abs[0:ro]**2).sum()

        for val in v_abs[ro:]:
            s1 += val 
            s2 += val**2
            if (s1-i*val)/sqrt(s2-2*s1*val+i*val**2) >= lam:
                i -= 1
                s1 -= val; s1 /= i
                s2 -= val**2; s2 /= i
                theta = s1-lam*sqrt((s2-s1**2)/(i-lam**2))
                break
            else:
                i += 1

        res = abs(v) - theta
        res[res<0] = 0
        return sign(v)*dot(abs(v), res)/dot(res,res)*res

