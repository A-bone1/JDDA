from LenetDANN import *
from DataLoader import *
from Utils import *
import scipy.io
import numpy as np
from tensorflow.contrib import slim
os.environ['CUDA_VISIBLE_DEVICES']='0'



class Train():
    def __init__(self,class_num,batch_size,iters,learning_rate,keep_prob,param):
        self.ClassNum=class_num

        self.BatchSize=batch_size
        self.Iters=iters
        self.LearningRate=learning_rate
        self.KeepProb=keep_prob
        self.target_loss_param=param[0]
        self.domain_loss_param=param[1]
        self.adver_loss_param=param[2]

        self.SourceData,self.SourceLabel=load_svhn('svhn')

        self.TargetData, self.TargetLabel=load_mnist('mnist')
        self.TestData, self.TestLabel = load_mnist('mnist',split='test')


        #######################################################################################
        self.source_image = tf.placeholder(tf.float32, shape=[self.BatchSize, 32,32,3],name="source_image")
        self.source_label = tf.placeholder(tf.float32, shape=[self.BatchSize, self.ClassNum],name="source_label")

        self.target_image = tf.placeholder(tf.float32, shape=[self.BatchSize, 32, 32,1],name="target_image")
        self.Training_flag = tf.placeholder(tf.bool, shape=None,name="Training_flag")
        self.l=tf.placeholder(tf.float32,[])



    def TrainNet(self):
        self.source_model=Lenet(inputs=self.source_image,training_flag=self.Training_flag, reuse=False,l=self.l)
        self.target_model=Lenet(inputs=self.target_image, training_flag=self.Training_flag,reuse=True,l=self.l)
        self.CalLoss()
        Slabel=np.tile([1,0],[self.BatchSize,1])
        Tlabel = np.tile([0, 1], [self.BatchSize, 1])
        self.advloss1=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.source_model.adv,labels=Slabel))
        self.advloss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.target_model.adv, labels=Tlabel))
        self.loss+=self.advloss1+self.advloss2






        varall=tf.trainable_variables()

        self.solver = tf.train.AdamOptimizer(learning_rate=self.LearningRate).minimize(self.loss)
        self.source_prediction = tf.argmax(self.source_model.softmax_output, 1)
        self.target_prediction = tf.argmax(self.target_model.softmax_output, 1)

        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            # self.solver = tf.train.AdamOptimizer(learning_rate=self.LearningRate).minimize(self.loss)
            init = tf.global_variables_initializer()
            sess.run(init)
            self.SourceLabel=sess.run(tf.one_hot(self.SourceLabel,10))
            self.TestLabel=sess.run(tf.one_hot(self.TestLabel,10))
            # self.source_model.weights_initial(sess)
            # self.target_model.weights_initial(sess)
            true_num = 0.0
            for step in range(self.Iters):
                # self.SourceData,self.SourceLabel=shuffle(self.SourceData,self.SourceLabel)
                p=float(step)/self.Iters
                l=2./(1. + np.exp(-10.*p))-1
                # print(l)
                i= step % int(self.SourceData.shape[0]/self.BatchSize)
                j= step % int(self.TargetData.shape[0]/self.BatchSize)
                source_batch_x = self.SourceData[i * self.BatchSize: (i + 1) * self.BatchSize]
                source_batch_y = self.SourceLabel[i * self.BatchSize: (i + 1) * self.BatchSize]
                target_batch_x = self.TargetData[j * self.BatchSize: (j + 1) * self.BatchSize]
                total_loss, source_loss, source_prediction,_= sess.run(
                    fetches=[self.loss, self.source_loss, self.source_prediction, self.solver],
                    feed_dict={self.source_image: source_batch_x, self.source_label: source_batch_y,self.target_image: target_batch_x, self.Training_flag: True,self.l:l})

                true_label = argmax(source_batch_y, 1)
                true_num = true_num + sum(true_label == source_prediction)

                # if step % 100==0:
                #     self.SourceData, self.SourceLabel = shuffle(self.SourceData, self.SourceLabel)
                if step % 200 ==0:
                    print "Iters-{} ### TotalLoss={} ### SourceLoss={} ###".format(step, total_loss, source_loss)
                    train_accuracy = true_num / (200*self.BatchSize)
                    true_num = 0.0
                    print " ########## train_accuracy={} ###########".format(train_accuracy)
                    self.Test(sess)
                if step % 2000 == 0:
                    # savedata = np.array(lossData)
                    # np.save("SVHNtoMNIST.npy", savedata)
                    # pass
                    # self.conputeTSNE(step, self.SourceData,  self.TargetData,self.SourceLabel, self.TargetLabel, sess)
                    self.SourceData,self.SourceLabel=shuffle(self.SourceData,self.SourceLabel)
                    # print("success")
            # savedata=np.array(lossData)
            # np.save("MNISTtoMNISTMSOU.npy",savedata)
    def CalLoss(self):
        self.source_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.source_label, logits=self.source_model.fc5)





    def CalLoss(self):
        self.source_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.source_label, logits=self.source_model.fc5)
        self.source_loss = tf.reduce_mean(self.source_cross_entropy)

        # self.CalTargetLoss(method="Entropy")
        # self.CalDomainLoss(method="MMD")
        # self.CalAdver()
        # self.L2Loss()
        # self.loss=self.source_loss+self.domain_loss_param*self.domain_loss
        self.loss=self.source_loss


    def L2Loss(self):
        all_variables = tf.trainable_variables()
        self.l2 = 1e-5 * tf.add_n([tf.nn.l2_loss(v) for v in all_variables if 'bias' not in v.name])

    def CalDomainLoss(self,method):
        if method=="MMD":
            Xs=self.source_model.fc4
            Xt=self.target_model.fc4
            diff=tf.reduce_mean(Xs, 0, keep_dims=False) - tf.reduce_mean(Xt, 0, keep_dims=False)
            self.domain_loss=tf.reduce_sum(tf.multiply(diff,diff))


        elif method=="KMMD":
            Xs=self.source_model.fc4
            Xt=self.target_model.fc4
            self.domain_loss=tf.maximum(0.0001,KMMD(Xs,Xt))



        elif method=="CORAL":
            Xs = self.source_model.fc4
            Xt = self.target_model.fc4
            # d=int(Xs.shape[1])
            # Xms = Xs - tf.reduce_mean(Xs, 0, keep_dims=True)
            # Xcs = tf.matmul(tf.transpose(Xms), Xms) / self.BatchSize
            # Xmt = Xt - tf.reduce_mean(Xt, 0, keep_dims=True)
            # Xct = tf.matmul(tf.transpose(Xmt), Xmt) / self.BatchSize
            # self.domain_loss = tf.reduce_sum(tf.multiply((Xcs - Xct), (Xcs - Xct)))
            # self.domain_loss=self.domain_loss / (4.0*d*d)
            self.domain_loss=self.coral_loss(Xs,Xt)


        elif method =='LCORAL':
            Xs = self.source_model.fc4
            Xt = self.target_model.fc4
            self.domain_loss=self.log_coral_loss(Xs,Xt)


    def CalTargetLoss(self,method):
        if method=="Entropy":
            trg_softmax=self.target_model.softmax_output
            self.target_loss=-tf.reduce_mean(tf.reduce_sum(trg_softmax * tf.log(trg_softmax), axis=1))


        elif method=="Manifold":
            pass




    def coral_loss(self, h_src, h_trg, gamma=1e-3):

        # regularized covariances (D-Coral is not regularized actually..)
        # First: subtract the mean from the data matrix
        batch_size = self.BatchSize
        h_src = h_src - tf.reduce_mean(h_src, axis=0)
        h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
        cov_source = (1. / (batch_size - 1)) * tf.matmul(h_src, h_src,
                                                         transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
        cov_target = (1. / (batch_size - 1)) * tf.matmul(h_trg, h_trg,
                                                         transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
        # Returns the Frobenius norm (there is an extra 1/4 in D-Coral actually)
        # The reduce_mean account for the factor 1/d^2
        return tf.reduce_mean(tf.square(tf.subtract(cov_source, cov_target)))

    def log_coral_loss(self, h_src, h_trg, gamma=1e-3):
        # regularized covariances result in inf or nan
        # First: subtract the mean from the data matrix
        batch_size = float(self.BatchSize)
        h_src = h_src - tf.reduce_mean(h_src, axis=0)
        h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
        cov_source = (1. / (batch_size - 1)) * tf.matmul(h_src, h_src,
                                                         transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
        cov_target = (1. / (batch_size - 1)) * tf.matmul(h_trg, h_trg,
                                                         transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
        # eigen decomposition
        eig_source = tf.self_adjoint_eig(cov_source)
        eig_target = tf.self_adjoint_eig(cov_target)
        log_cov_source = tf.matmul(eig_source[1],
                                   tf.matmul(tf.diag(tf.log(eig_source[0])), eig_source[1], transpose_b=True))
        log_cov_target = tf.matmul(eig_target[1],
                                   tf.matmul(tf.diag(tf.log(eig_target[0])), eig_target[1], transpose_b=True))

        # Returns the Frobenius norm
        return tf.reduce_mean(tf.square(tf.subtract(log_cov_source, log_cov_target)))

    # ~ return tf.reduce_mean(tf.reduce_max(eig_target[0]))
    # ~ return tf.to_float(tf.equal(tf.count_nonzero(h_src), tf.count_nonzero(h_src)))


    def Test(self,sess):
        true_num=0.0
        # num=int(self.TargetData.shape[0]/self.BatchSize)
        num = int(self.TestData.shape[0] / self.BatchSize)
        total_num=num*self.BatchSize
        for i in range (num):
            # self.TestData, self.TestLabel = shuffle(self.TestData, self.TestLabel)
            k = i % int(self.TestData.shape[0] / self.BatchSize)
            target_batch_x = self.TestData[k * self.BatchSize: (k + 1) * self.BatchSize]
            target_batch_y= self.TestLabel[k * self.BatchSize: (k + 1) * self.BatchSize]
            prediction=sess.run(fetches=self.target_prediction, feed_dict={self.target_image:target_batch_x, self.Training_flag: False})
            true_label = argmax(target_batch_y, 1)

            true_num+=sum(true_label==prediction)
        accuracy=true_num / total_num
        print "###########  Test Accuracy={} ##########".format(accuracy)

def main():
    target_loss_param =0
    domain_loss_param =0.05
    adver_loss_param=0
    param=[target_loss_param, domain_loss_param,adver_loss_param]
    Runer=Train(class_num=10,batch_size=128,iters=200000,learning_rate=0.0001,keep_prob=1,param=param)
    Runer.TrainNet()

def load_mnist(image_dir, split='train'):
    print ('Loading MNIST dataset.')

    image_file = 'train.pkl' if split == 'train' else 'test.pkl'
    image_dir = os.path.join(image_dir, image_file)
    with open(image_dir, 'rb') as f:
        mnist = pickle.load(f)
    images = mnist['X'] / 127.5 - 1
    labels = mnist['y']
    labels=np.squeeze(labels).astype(int)

    return images,labels
def load_svhn(image_dir, split='train'):
    print ('Loading SVHN dataset.')

    image_file = 'train_32x32.mat' if split == 'train' else 'test_32x32.mat'

    image_dir = os.path.join(image_dir, image_file)
    svhn = scipy.io.loadmat(image_dir)
    images = np.transpose(svhn['X'], [3, 0, 1, 2]) / 127.5 - 1
    # ~ images= resize_images(images)
    labels = svhn['y'].reshape(-1)
    labels[np.where(labels == 10)] = 0
    return images, labels

def load_USPS(image_dir,split='train'):
    print('Loading USPS dataset.')
    image_file='USPS_train.pkl' if split=='train' else 'USPS_test.pkl'
    image_dir=os.path.join(image_dir,image_file)
    with open(image_dir, 'rb') as f:
        usps = pickle.load(f)
    images = usps['data']
    images=np.reshape(images,[-1,32,32,1])
    labels = usps['label']
    labels=np.squeeze(labels).astype(int)
    return images,labels

def load_syn(image_dir,split='train'):
    print('load syn dataset')
    image_file='synth_train_32x32.mat' if split=='train' else 'synth_test_32x32.mat'
    image_dir=os.path.join(image_dir,image_file)
    syn = scipy.io.loadmat(image_dir)
    images = np.transpose(syn['X'], [3, 0, 1, 2]) / 127.5 - 1
    labels = syn['y'].reshape(-1)
    return images,labels


def load_mnistm(image_dir,split='train'):
    print('Loading mnistm dataset.')
    image_file='mnistm_train.pkl' if split=='train' else 'mnistm_test.pkl'
    image_dir=os.path.join(image_dir,image_file)
    with open(image_dir, 'rb') as f:
        mnistm = pickle.load(f)
    images = mnistm['data']

    labels = mnistm['label']
    labels=np.squeeze(labels).astype(int)
    return images,labels

if __name__=="__main__":
    main()
