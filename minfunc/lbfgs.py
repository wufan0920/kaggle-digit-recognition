import numpy as np
import math
import gc

def lbfgs_update(y,s,old_dirs,old_stps,k,H):
    ys = y.T.dot(s)
    if ys > 1e-10:
        numCorrections = len(old_dirs)
        if numCorrections < k:
            #Full Update
            old_dirs.append(s)
            old_stps.append(y) 
        else:
            #Limited-Memory Update
            tmp_dirs = old_dirs[1:]
            tmp_stps = old_stps[1:]
            tmp_dirs.append(s)
            tmp_stps.append(y)            
            old_dirs = tmp_dirs 
            old_stps = tmp_stps 

        #Update scale of initial Hessian approximation
        param = ys/(y.T.dot(y))
        Hdiag = param
    else:
        print 'ng'
        Hdiag = H 

    return Hdiag,old_dirs,old_stps

def lbfgs_descent(g,s,y,H):
    k = len(s)
    p = len(s[0])

    ro = np.zeros((k,1))
    q  = np.zeros((p,k+1))
    r  = np.zeros((p,k+1))
    al = np.zeros((k,1))
    be = np.zeros((k,1))

    for i in range(k):
        ro[i] = 1/(y[i].dot(s[i]))

    q[:,k] = g

    for i in range(k-1,-1,-1):
        al[i] = (ro[i]*s[i]).dot(q[:,i+1])
        q[:,i] = q[:,i+1] - al[i]*y[i]

    r[:,0] = H*q[:,0]

    for i in range(k):
        be[i] = ro[i]*y[i].dot(r[:,i])
        r[:,i+1] = r[:,i] + s[i]*(al[i]-be[i])

    d = r[:,k]
    return d

def lbfgs(data,theta,grad_func,validator,max_iter=400,k=100,alpha=0.3,batch=1):

    for itr in range(max_iter):
        print itr
        batch_size = data.shape[1]/batch
        for i in range(batch):
            grad = grad_func(data[:,i*batch_size:(i+1)*batch_size],theta)
            if itr == 0:
                d = -grad
                old_grad = 0
                old_dirs = []
                old_stps = []
                H = 1
            else:
                H,old_dirs,old_stps = lbfgs_update(grad-old_grad,alpha*d,old_dirs,old_stps,k,H)
                d = lbfgs_descent(-grad,old_dirs,old_stps,H)

            theta = theta+alpha*d
            old_grad = grad

        validator.validate(theta)
        if validator.early_stop() == True:
            break

    print 'validation accuracy:', validator.get_accuracy()
    return validator.get_result()

def lbfgs_without_validate(data,theta,grad_func,max_iter=400,k=100,alpha=0.3,batch=1):

    for itr in range(max_iter):
        print itr
        batch_size = data.shape[1]/batch
        for i in range(batch):
            grad = grad_func(data[:,i*batch_size:(i+1)*batch_size],theta)
            if itr == 0:
                d = -grad
                old_grad = 0
                old_dirs = []
                old_stps = []
                H = 1
            else:
                H,old_dirs,old_stps = lbfgs_update(grad-old_grad,alpha*d,old_dirs,old_stps,k,H)
                d = lbfgs_descent(-grad,old_dirs,old_stps,H)

            theta = theta+alpha*d
            old_grad = grad

    return theta
