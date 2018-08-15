def train_value():
    value_path = os.path.join(FLAGS.pre_D_model_dir, '..', 'value')
    if not os.path.exists(value_path):
        os.makedirs(value_path)
    def build_summaries():
        loss = tf.Variable(0.)
        tf.summary.scalar("V-loss", loss)
        summary_vars = [loss]
        summary_ops = tf.summary.merge_all()
        return summary_ops, summary_vars
    # parse data and build vocab if there do not exist one.
    train, _, _ = data_utils.prepare_data(FLAGS.data_dir, FLAGS.data_path, FLAGS.vocab_size)
    with tf.Session() as sess:
        # build the model
        model = Seq2Seq(
            'TRAIN',
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.vocab_size,
            _buckets,
            FLAGS.lr,
            FLAGS.lr_decay,
            FLAGS.grad_norm,
            critic=FLAGS.gan_type,
            critic_size=FLAGS.gan_size,
            critic_num_layers=FLAGS.gan_num_layers,
            other_option=FLAGS.option,
            use_attn=FLAGS.use_attn,
            output_sample=True,
            input_embed=True,
            feed_prev=False,
            batch_size=FLAGS.batch_size,
            D_lr=FLAGS.D_lr,
            D_lr_decay_factor=FLAGS.D_lr_decay_factor,
            dtype=tf.float32)
        # build summary and initialize
        summary_ops, summary_vars = build_summaries()
        sess.run(tf.variables_initializer(tf.global_variables()))
        log_dir = os.path.join(value_path, 'log')
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        pre_V_ckpt = tf.train.get_checkpoint_state(value_path)
        if pre_V_ckpt and tf.train.checkpoint_exists(pre_V_ckpt.model_checkpoint_path):
            print ('read in model from {}'.format(pre_V_ckpt.model_checkpoint_path))
            model.pre_value_saver.restore(sess, pre_V_ckpt.model_checkpoint_path)
        else:
            print ('no previous model, create a new one')
        pre_ckpt = tf.train.get_checkpoint_state(FLAGS.pre_model_dir)
        if pre_ckpt and tf.train.checkpoint_exists(pre_ckpt.model_checkpoint_path):
            print ('read in model from {}'.format(pre_ckpt.model_checkpoint_path))
            model.pre_saver.restore(sess, pre_ckpt.model_checkpoint_path)
        pre_D_ckpt = tf.train.get_checkpoint_state(FLAGS.pre_D_model_dir)
        if pre_D_ckpt and tf.train.checkpoint_exists(pre_D_ckpt.model_checkpoint_path):
            print ('read in model from {}'.format(pre_D_ckpt.model_checkpoint_path))
            model.pre_D_saver.restore(sess, pre_D_ckpt.model_checkpoint_path)

        # load in train and dev(valid) data with buckets
        train_set = read_data_with_buckets(train, FLAGS.max_train_data_size)
        train_buckets_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        train_total_size = float(sum(train_buckets_sizes))
        print ('each buckets has: {d}'.format(d=train_buckets_sizes))
        train_buckets_scale = [sum(train_buckets_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_buckets_sizes))]

        # main process
        step_time, loss = 0.0, 0.0
        current_step = 0
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])
            encoder_inputs, decoder_inputs, weights, seq_lens, _ = \
                get_batch_with_buckets(train_set, FLAGS.batch_size, bucket_id)

            start_time = time.time()
            step_loss, _ = \
                model.train_step(sess, encoder_inputs, \
                                 decoder_inputs, weights, \
                                 bucket_id, seq_lens, GAN_mode='V')
            loss += step_loss / FLAGS.steps_per_checkpoint
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            
            current_step += 1
            if current_step % FLAGS.steps_per_checkpoint == 0:
                print("global step %d; step-time %.2f; V-loss %.4f"
                      % (model.global_V_step.eval(),
                         step_time, loss))
                feed_dict = {}
                feed_dict[summary_vars[0]] = loss
                summary_str = sess.run(summary_ops,
                                       feed_dict=feed_dict)
                writer.add_summary(summary_str, model.global_V_step.eval())
                writer.flush()

                ckpt_path = os.path.join(value_path, "ckpt")
                model.pre_value_saver.save(sess, ckpt_path, global_step=model.global_V_step)
                if model.global_V_step.eval() >= 10000:
                    return

                step_time, loss = 0.0, 0.0

                sys.stdout.flush()

def train_critic():
    if not os.path.exists(FLAGS.pre_D_model_dir):
        os.makedirs(FLAGS.pre_D_model_dir)
    def build_summaries():
        D_loss = tf.Variable(0.)
        tf.summary.scalar("D_loss", D_loss)
        summary_vars = [D_loss]
        summary_ops = tf.summary.merge_all()
        return summary_ops, summary_vars
    # parse data and build vocab if there do not exist one.
    train, _, _ = data_utils.prepare_data(FLAGS.data_dir, FLAGS.data_path, FLAGS.vocab_size)
    with tf.Session() as sess:
        # build the model
        model = Seq2Seq(
            'TRAIN',
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.vocab_size,
            _buckets,
            FLAGS.lr,
            FLAGS.lr_decay,
            FLAGS.grad_norm,
            critic=FLAGS.gan_type,
            critic_size=FLAGS.gan_size,
            critic_num_layers=FLAGS.gan_num_layers,
            other_option=FLAGS.option,
            use_attn=FLAGS.use_attn,
            output_sample=True,
            input_embed=True,
            feed_prev=False,
            batch_size=FLAGS.batch_size,
            D_lr=FLAGS.D_lr,
            D_lr_decay_factor=FLAGS.D_lr_decay_factor,
            dtype=tf.float32)
        # build summary and initialize
        summary_ops, summary_vars = build_summaries()
        sess.run(tf.variables_initializer(tf.global_variables()))
        log_dir = os.path.join(FLAGS.pre_D_model_dir, 'log')
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        pre_D_ckpt = tf.train.get_checkpoint_state(FLAGS.pre_D_model_dir)
        if pre_D_ckpt and tf.train.checkpoint_exists(pre_D_ckpt.model_checkpoint_path):
            print ('read in model from {}'.format(pre_D_ckpt.model_checkpoint_path))
            model.pre_D_saver.restore(sess, pre_D_ckpt.model_checkpoint_path)
        else:
            print ('no previous model, create a new one')
        
        pre_ckpt = tf.train.get_checkpoint_state(FLAGS.pre_model_dir)
        if pre_ckpt and tf.train.checkpoint_exists(pre_ckpt.model_checkpoint_path):
            print ('read in model from {}'.format(pre_ckpt.model_checkpoint_path))
            model.pre_saver.restore(sess, pre_ckpt.model_checkpoint_path)

        # load in train and dev(valid) data with buckets
        train_set = read_data_with_buckets(train, FLAGS.max_train_data_size)
        train_buckets_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        train_total_size = float(sum(train_buckets_sizes))
        print ('each buckets has: {d}'.format(d=train_buckets_sizes))
        train_buckets_scale = [sum(train_buckets_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_buckets_sizes))]

        # main process
        step_time, D_loss = 0.0, 0.0
        current_step = 0
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])
            encoder_inputs, decoder_inputs, weights, seq_lens, _ = \
                get_batch_with_buckets(train_set, FLAGS.batch_size, bucket_id)

            start_time = time.time()
            step_D_loss, _ = \
                model.train_step(sess, encoder_inputs, \
                                 decoder_inputs, weights, \
                                 bucket_id, seq_lens, GAN_mode='D')
            D_loss += step_D_loss / FLAGS.steps_per_checkpoint
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            
            current_step += 1
            if current_step % FLAGS.steps_per_checkpoint == 0:
                print("global step D %d; step-time %.2f; D-loss %.4f"
                      % (model.global_D_step.eval(),
                         step_time, D_loss))
                feed_dict = {}
                feed_dict[summary_vars[0]] = D_loss
                summary_str = sess.run(summary_ops,
                                       feed_dict=feed_dict)
                writer.add_summary(summary_str, model.global_D_step.eval())
                writer.flush()

                D_ckpt_path = os.path.join(FLAGS.pre_D_model_dir, "ckpt")
                model.pre_D_saver.save(sess, D_ckpt_path, global_step=model.global_D_step)
                if model.global_D_step.eval() >= 20000:
                    return
                step_time, D_loss = 0.0, 0.0
                sys.stdout.flush()
