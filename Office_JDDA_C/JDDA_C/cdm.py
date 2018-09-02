import random
import tensorflow as tf
from functools import partial



def mmatch(x1,x2,n_moments):
    mx1=tf.reduce_mean(x1,axis=0)
    mx2=tf.reduce_mean(x2,axis=0)
    sx1=x1-mx1
    sx2=x2-mx2
    dm=matchnorm(mx1,mx2)
    scms=dm
    for i in range(n_moments-1):
        scms+=scm(sx1,sx2,i+2)
    return scms

def matchnorm(x1,x2):
    return tf.sqrt(tf.reduce_sum((x1-x2)**2))

def scm(sx1,sx2,k):

    ss1=tf.reduce_mean(sx1**k,axis=0)
    ss2=tf.reduce_mean(sx2**k,axis=0)
    return matchnorm(ss1,ss2)