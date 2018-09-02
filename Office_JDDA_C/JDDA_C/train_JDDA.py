import os, sys
import numpy as np
import tensorflow as tf
import datetime
from model import ResNetModel
sys.path.insert(0, '../utils')
from preprocessor import BatchPreprocessor
os.environ['CUDA_VISIBLE_DEVICES']='0'
from Utils import *
from center_loss import *
from cdm import *
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
UPDATE_OPS_COLLECTION = 'resnet_update_ops'
FC_WEIGHT_STDDEV = 0.01


tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for adam optimizer')
tf.app.flags.DEFINE_integer('resnet_depth', 50, 'ResNet architecture to be used: 50, 101 or 152')
tf.app.flags.DEFINE_integer('num_epochs',401, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('num_classes',31, 'Number of classes')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_string('train_layers', "fc,scale5/block3", 'Finetuning layers, seperated by commas')
tf.app.flags.DEFINE_string('multi_scale', '', 'As preprocessing; scale the image randomly between 2 numbers and crop randomly at network\'s input size')
tf.app.flags.DEFINE_string('training_file', '../data/amazon.txt', 'Training dataset file')
tf.app.flags.DEFINE_string('val_file', '../data/webcam.txt', 'Validation dataset file')
tf.app.flags.DEFINE_string('tensorboard_root_dir', '../training', 'Root directory to put the training logs and weights')
tf.app.flags.DEFINE_integer('log_step', 10, 'Logging period in terms of iteration')

FLAGS = tf.app.flags.FLAGS


def main(_):
    # Create training directories
    now = datetime.datetime.now()
    train_dir_name = now.strftime('resnet_%Y%m%d_%H%M%S')
    train_dir = os.path.join(FLAGS.tensorboard_root_dir, train_dir_name)
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    tensorboard_dir = os.path.join(train_dir, 'tensorboard')
    tensorboard_train_dir = os.path.join(tensorboard_dir, 'train')
    tensorboard_val_dir = os.path.join(tensorboard_dir, 'val')

    if not os.path.isdir(FLAGS.tensorboard_root_dir): os.mkdir(FLAGS.tensorboard_root_dir)
    if not os.path.isdir(train_dir): os.mkdir(train_dir)
    if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
    if not os.path.isdir(tensorboard_dir): os.mkdir(tensorboard_dir)
    if not os.path.isdir(tensorboard_train_dir): os.mkdir(tensorboard_train_dir)
    if not os.path.isdir(tensorboard_val_dir): os.mkdir(tensorboard_val_dir)

    # Write flags to txt
    flags_file_path = os.path.join(train_dir, 'flags.txt')
    flags_file = open(flags_file_path, 'w')
    flags_file.write('learning_rate={}\n'.format(FLAGS.learning_rate))
    flags_file.write('resnet_depth={}\n'.format(FLAGS.resnet_depth))
    flags_file.write('num_epochs={}\n'.format(FLAGS.num_epochs))
    flags_file.write('batch_size={}\n'.format(FLAGS.batch_size))
    flags_file.write('train_layers={}\n'.format(FLAGS.train_layers))
    flags_file.write('multi_scale={}\n'.format(FLAGS.multi_scale))
    flags_file.write('tensorboard_root_dir={}\n'.format(FLAGS.tensorboard_root_dir))
    flags_file.write('log_step={}\n'.format(FLAGS.log_step))
    flags_file.close()

    # Placeholders
    source = tf.placeholder(tf.float32, [FLAGS.batch_size, 224, 224, 3])
    target = tf.placeholder(tf.float32, [FLAGS.batch_size, 224, 224, 3])
    y = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
    is_training = tf.placeholder('bool', [])

    par=tf.Variable(tf.constant(0.2),dtype=tf.float32)

    # Model
    train_layers = FLAGS.train_layers.split(',')
    source_model = ResNetModel(source,is_training, depth=FLAGS.resnet_depth, num_classes=FLAGS.num_classes)
    target_model = ResNetModel(target,is_training,reuse=True, depth=FLAGS.resnet_depth, num_classes=FLAGS.num_classes)


    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=source_model.prob, labels=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([cross_entropy_mean] + regularization_losses)
    # domain_loss=tf.maximum(0.0001,KMMD(source_model.avg_pool,target_model.avg_pool))
    domain_loss=coral_loss(source_model.avg_pool,target_model.avg_pool)
    centers_update_op,discriminative_loss=CenterBased(source_model.avg_pool,y)
    # domain_loss = mmatch(source_model.avg_pool,target_model.avg_pool, 5)
    # domain_loss = log_coral_loss(source_model.adapt, target_model.adapt)
    loss=loss+1*par*domain_loss+0.03*discriminative_loss





    # train_op = model.optimize(FLAGS.learning_rate, train_layers)
    Varall=tf.trainable_variables()
    # print(Varall)
    trainable_var_names = ['weights', 'biases', 'beta', 'gamma']
    var_list = [v for v in tf.trainable_variables() if
                v.name.split(':')[0].split('/')[-1] in trainable_var_names and
                contains(v.name, train_layers)]
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)#.minimize(loss, var_list=var_list)

    # ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    # tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss]))

    # batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    # batchnorm_updates_op = tf.group(*batchnorm_updates)
    # train_op=tf.group(train_op, batchnorm_updates_op)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # total_op=tf.group(update_ops,centers_update_op)
    with tf.control_dependencies(update_ops):
        with tf.control_dependencies([centers_update_op]):
            train_op = optimizer.minimize(loss,var_list=var_list)

    # Training accuracy of the model
    correct_pred = tf.equal(tf.argmax(source_model.prob, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Summaries
    tf.summary.scalar('train_loss', loss)
    tf.summary.scalar('train_accuracy', accuracy)
    merged_summary = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(tensorboard_train_dir)
    val_writer = tf.summary.FileWriter(tensorboard_val_dir)
    saver = tf.train.Saver()

    # Batch preprocessors
    multi_scale = FLAGS.multi_scale.split(',')
    if len(multi_scale) == 2:
        multi_scale = [int(multi_scale[0]), int(multi_scale[1])]
    else:
        multi_scale = None

    train_preprocessor = BatchPreprocessor(dataset_file_path=FLAGS.training_file, num_classes=FLAGS.num_classes,
                                           output_size=[224, 224], horizontal_flip=False, shuffle=True, multi_scale=multi_scale)

    target_preprocessor = BatchPreprocessor(dataset_file_path='../data/webcam.txt', num_classes=FLAGS.num_classes,output_size=[224, 224],shuffle=True)

    val_preprocessor = BatchPreprocessor(dataset_file_path=FLAGS.val_file, num_classes=FLAGS.num_classes, output_size=[224, 224])

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = np.floor(len(train_preprocessor.labels) / FLAGS.batch_size).astype(np.int16)
    target_batches_per_epoch = np.floor(len(target_preprocessor.labels) / FLAGS.batch_size).astype(np.int16)
    val_batches_per_epoch = np.floor(len(val_preprocessor.labels) / FLAGS.batch_size).astype(np.int16)

    # train_batches_per_epoch=np.minimum(train_batches_per_epoch,target_batches_per_epoch)
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        varall=tf.trainable_variables()

        sess.run(tf.global_variables_initializer())
        train_writer.add_graph(sess.graph)

        # Load the pretrained weights
        source_model.load_original_weights(sess, skip_layers=train_layers)
        # target_model.load_original_weights(sess, skip_layers=train_layers)

        # Directly restore (your model should be exactly the same with checkpoint)
        # saver.restore(sess, "/Users/dgurkaynak/Projects/marvel-training/alexnet64-fc6/model_epoch10.ckpt")

        print("{} Start training...".format(datetime.datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.datetime.now(), tensorboard_dir))

        for epoch in range(FLAGS.num_epochs):
            print("{} Epoch number: {}".format(datetime.datetime.now(), epoch+1))
            step = 1
            param=2/(1+np.exp(-10*(epoch)/FLAGS.num_epochs))-1
            print(param)
            sess.run(tf.assign(par,param))
            print(sess.run(par))

            # Start training
            while step < train_batches_per_epoch:
                if step%target_batches_per_epoch==0:
                    target_preprocessor.reset_pointer()
                batch_xs, batch_ys = train_preprocessor.next_batch(FLAGS.batch_size)
                batch_xt, batch_yt = target_preprocessor.next_batch(FLAGS.batch_size)
                sess.run(train_op, feed_dict={source: batch_xs,target:batch_xt, y: batch_ys, is_training: True})

                # Logging
                # if step % FLAGS.log_step == 0:
                #     s = sess.run(merged_summary, feed_dict={source: batch_xs, y: batch_ys, is_training: False})
                #     train_writer.add_summary(s, epoch * train_batches_per_epoch + step)

                step += 1

            # Epoch completed, start validation
            print("{} Start validation".format(datetime.datetime.now()))
            test_acc = 0.
            test_count = 0

            for _ in range(val_batches_per_epoch):
                batch_tx, batch_ty = val_preprocessor.next_batch(FLAGS.batch_size)
                acc = sess.run(accuracy, feed_dict={source: batch_tx, y: batch_ty, is_training: False})
                test_acc += acc
                test_count += 1

            test_acc /= test_count
            s = tf.Summary(value=[
                tf.Summary.Value(tag="validation_accuracy", simple_value=test_acc)
            ])
            val_writer.add_summary(s, epoch+1)
            print("{} Validation Accuracy = {:.4f}".format(datetime.datetime.now(), test_acc))

            # Reset the dataset pointers
            val_preprocessor.reset_pointer()
            train_preprocessor.reset_pointer()
            target_preprocessor.reset_pointer()
            #
            # print("{} Saving checkpoint of model...".format(datetime.datetime.now()))
            #
            # #save checkpoint of the model
            # checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch'+str(epoch+1)+'.ckpt')
            # save_path = saver.save(sess, checkpoint_path)
            #
            # print("{} Model checkpoint saved at {}".format(datetime.datetime.now(), checkpoint_path))

def contains(target_str, search_arr):
    rv = False

    for search_str in search_arr:
        if search_str in target_str:
            rv = True
            break

    return rv

def coral_loss(h_src,h_trg,batchsize=128, gamma=1e-3):

    # regularized covariances (D-Coral is not regularized actually..)
    # First: subtract the mean from the data matrix
    batch_size = batchsize
    h_src = h_src - tf.reduce_mean(h_src, axis=0)
    h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
    cov_source = (1. / (batch_size - 1)) * tf.matmul(h_src, h_src,
                                                     transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
    cov_target = (1. / (batch_size - 1)) * tf.matmul(h_trg, h_trg,
                                                     transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
    # Returns the Frobenius norm (there is an extra 1/4 in D-Coral actually)
    # The reduce_mean account for the factor 1/d^2
    return tf.reduce_mean(tf.square(tf.subtract(cov_source, cov_target)))

def MMD(xs,xt):
    diff = tf.reduce_mean(xs, 0, keep_dims=False) - tf.reduce_mean(xt, 0, keep_dims=False)
    test=tf.multiply(diff, diff)
    loss=tf.reduce_sum(tf.multiply(diff, diff))
    return tf.reduce_sum(tf.multiply(diff, diff))


def MMD1(xs,xt):
    diff=xs-xt
    test=tf.matmul(diff,tf.transpose(diff))
    loss=tf.reduce_mean(tf.matmul(diff,tf.transpose(diff)))
    return loss

def log_coral_loss(h_src, h_trg,batch_size=128,gamma=1e-3):
    # regularized covariances result in inf or nan
    # First: subtract the mean from the data matrix
    batch_size = float(batch_size)
    h_src = h_src - tf.reduce_mean(h_src, axis=0)
    h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
    cov_source = (1. / (batch_size - 1)) * tf.matmul(h_src, h_src,
                                                     transpose_a=True)  + gamma * tf.eye(128)
    cov_target = (1. / (batch_size - 1)) * tf.matmul(h_trg, h_trg,
                                                     transpose_a=True)  + gamma * tf.eye(128)
    # eigen decomposition
    eig_source = tf.self_adjoint_eig(cov_source)
    eig_target = tf.self_adjoint_eig(cov_target)
    log_cov_source = tf.matmul(eig_source[1],
                               tf.matmul(tf.diag(tf.log(eig_source[0])), eig_source[1], transpose_b=True))
    log_cov_target = tf.matmul(eig_target[1],
                               tf.matmul(tf.diag(tf.log(eig_target[0])), eig_target[1], transpose_b=True))

    # Returns the Frobenius norm
    return tf.reduce_mean(tf.square(tf.subtract(log_cov_source, log_cov_target)))

def CenterBased(fc4,y):
    source_label=y
    Xs = fc4
    labels = tf.argmax(source_label, 1)
    inter_loss, intra_loss, centers_update_op = get_center_loss(Xs, labels, 0.5, 31)
    discriminative_loss = intra_loss + inter_loss
    discriminative_loss = discriminative_loss / (31 * 128 + 31 * 31)
    return centers_update_op,discriminative_loss

if __name__ == '__main__':
    tf.app.run()
