from src.configuration.coco_conf.make_dataset import CocoDataset
from src.configuration.base_config import Config
from make_ground_truth import GroundTruth
import tensorflow as tf
import time
from src.MYHRNet import HRNet, compute_loss

if __name__ == '__main__':
    # GPU settings
    gpu_divice = '/gpu:0'
    cfg = Config()
    # hrnet = get_model(cfg)
    # print_model_summary(hrnet)
    # if tf.test.gpu_device_name():
    #     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    # else:
    #     print("Please install GPU version of TF")
    # Dataset
    coco = CocoDataset(config_params=cfg, dataset_type="train")
    dataset = coco.generate_dataset()
    ite = dataset.make_initializable_iterator()

    global_step = tf.Variable(0, trainable=False)
    input_images = tf.placeholder(tf.float32, [None, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, cfg.CHANNELS])
    ground_truth = tf.placeholder(tf.float32, [None, cfg.HEATMAP_HEIGHT, cfg.HEATMAP_WIDTH, cfg.num_of_joints])
    #input_images = tf.cast(input_images / 255.0, tf.float32, name='change_type')

    net_output = HRNet(input=input_images, is_training=True)
    loss = compute_loss(net_output=net_output, ground_truth=ground_truth)
    saver = tf.train.Saver()
    train_step = tf.train.AdamOptimizer(cfg.LEARNING_RATE).minimize(loss, global_step=global_step)


    with tf.Session() as sess:
        #with tf.device(gpu_divice):
            sess.run(tf.global_variables_initializer())
            #sess.run(tf.local_variables_initializer())
            for epoch in range(cfg.EPOCHS):
                epoch_time = time.time()
                # print(dataset.output_shapes)
                sess.run(ite.initializer)
                step = 0
                while True:
                    try:
                        batch_data = sess.run(ite.get_next())
                        #print(batch_data)
                        gt = GroundTruth(cfg, batch_data)
                        step += 1
                        images, target, target_weight = gt.get_ground_truth()
                        sess.run(images)
                        sess.run(target)
                        print(images)
                        print(target)
                        train_step.run(feed_dict={input_images: images.eval(), ground_truth: target.eval()})
                        tloss, tnet_output = sess.run([loss, net_output],
                                                      feed_dict={input_images: images.eval(),
                                                                 ground_truth: target.eval()})
                        print('Epoch {:>2}/{}, step = {:>6}/{:>6}, loss = {:.6f}, time = {}'
                              .format(epoch, cfg.EPOCHS, step, int(118287 / cfg.BATCH_SIZE), tloss,
                                      time.time() - epoch_time))
                        # 118287 is not exact
                    except tf.errors.OutOfRangeError:
                        break
                if epoch % cfg.SAVE_FREQUENCY == 0:
                    saver.save(sess, cfg.save_weights_dir + 'epoch{}.ckpt'.format(epoch), global_step=global_step)
                    print('Model saved in: {}'.format(cfg.save_weights_dir + 'epoch{}.ckpt'.format(epoch)))


