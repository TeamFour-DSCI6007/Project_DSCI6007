"""TB-Net evaluation."""

import os
import argparse
# import moxing as mox
import math

from mindspore import context, Model, load_checkpoint, load_param_into_net
import mindspore.common.dtype as mstype

from src import tbnet, config, metrics, dataset


def get_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(description='Train TBNet.')

    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        default='steam',
        help="'steam' dataset is supported currently"
    )

    parser.add_argument(
        '--csv',
        type=str,
        required=False,
        default='test.csv',
        help="the csv datafile inside the dataset folder (e.g. test.csv)"
    )

    parser.add_argument(
        '--checkpoint_id',
        type=int,
        required=False,
        default=19,
        help="use which checkpoint(.ckpt) file to eval"
    )

    parser.add_argument(
        '--device_id',
        type=int,
        required=False,
        default=0,
        help="device id"
    )

    parser.add_argument(
        '--device_target',
        type=str,
        required=False,
        default='GPU',
        choices=['GPU', 'Ascend'],
        help="run code on GPU or Ascend NPU"
    )

    parser.add_argument(
        '--data_url',
        type=str,
        default="./Data",
        help='path where the dataset is saved'
    )

    parser.add_argument(
        '--ckpt_url',
        help='model to save/load',
        default='./ckpt_url'
    )

    parser.add_argument(
        '--result_url',
        help='result folder to save/load',
        default='./result'
    )

    parser.add_argument(
        '--run_mode',
        type=str,
        required=False,
        default='graph',
        choices=['graph', 'pynative'],
        help="run code by GRAPH mode or PYNATIVE mode"
    )

    return parser.parse_args()


def eval_tbnet():
    """Evaluation process."""
    print("starting evaluation")
    args = get_args()
    print("starting evaluation")
    # home = os.path.dirname(os.path.realpath(__file__))
    home = "tb-net-latest"
    obs_data_url = args.data_url
    args.data_url = home
    if not os.path.exists(args.data_url):
        os.mkdir(args.data_url)
    # try:
    #     mox.file.copy_parallel(obs_data_url, args.data_url)
    #     print("Successfully Download {} to {}".format(obs_data_url,
    #                                                   args.data_url))
    # except Exception as e:
    #     print('moxing download {} to {} failed: '.format(
    #         obs_data_url, args.data_url) + str(e))

    os.system("python " + "preprocess_dataset.py")

    obs_ckpt_url = args.ckpt_url
    args.ckpt_url =  'checkpoints/tbnet_epoch' + str(args.checkpoint_id) + '.ckpt'
    # try:
    #     # mox.file.copy(obs_ckpt_url, args.ckpt_url)
    #     print("Successfully Download {} to {}".format(obs_ckpt_url,
    #                                                   args.ckpt_url))
    # except Exception as e:
    #     print('moxing download {} to {} failed: '.format(
    #         obs_ckpt_url, args.ckpt_url) + str(e))

    obs_result_url = args.result_url
    args.result_url = "results"
    if not os.path.exists(args.result_url):
        os.mkdir(args.result_url)

    # config_path = os.path.join(home, 'data', args.dataset, 'config.json')
    config_path = "data\steam\config.json"
    # test_csv_path = os.path.join(home, 'data', args.dataset, args.csv)
    test_csv_path = r"data\steam\test.csv"
    ckpt_path = os.path.join(home, 'checkpoints')

    context.set_context(device_id=args.device_id)
    if args.run_mode == 'graph':
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

    print(f"creating dataset from {test_csv_path}...")
    net_config = config.TBNetConfig(config_path)
    if args.device_target == 'CPU':
        net_config.per_item_paths = math.ceil(net_config.per_item_paths / 16) * 16
        net_config.embedding_dim = math.ceil(net_config.embedding_dim / 16) * 16
    eval_ds = dataset.create(test_csv_path, net_config.per_item_paths, train=True).batch(net_config.batch_size)

    print(f"creating TBNet from checkpoint {args.checkpoint_id} for evaluation...")
    network = tbnet.TBNet(net_config)
    if args.device_target == 'CPU':
        network.to_float(mstype.float16)
    param_dict = load_checkpoint(args.ckpt_url)
    load_param_into_net(network, param_dict)

    loss_net = tbnet.NetWithLossClass(network, net_config)
    train_net = tbnet.TrainStepWrap(loss_net, net_config.lr)
    train_net.set_train()
    eval_net = tbnet.PredictWithSigmoid(network)
    model = Model(network=train_net, eval_network=eval_net, metrics={'auc': metrics.AUC(), 'acc': metrics.ACC()})
    # model.build(valid_dataset=eval_ds, epoch=1)

    print("evaluating...")
    e_out = model.eval(eval_ds)
    print(f'Test AUC:{e_out ["auc"]} ACC:{e_out ["acc"]}')
    filename = 'result.txt'
    file_path = os.path.join(args.result_url, filename)
    with open(file_path, 'a+') as file:
        file.write(f'Test AUC:{e_out["auc"]} ACC:{e_out["acc"]}')

    from src.aggregator import InferenceAggregator
    from src import steam
    # data_path = os.path.join(home, 'data', args.dataset, 'infer.csv') 
    data_path = r"data\steam\infer.csv"
    # translate_path = os.path.join(home, 'data', args.dataset, 'translate.json')
    translate_path = r"data\steam\translate.json"
    print(f"creating dataset from {data_path}...")
    infer_ds = dataset.create(data_path, net_config.per_item_paths, train=False)
    # for x in infer_ds:
    #     print(x)
    infer_ds = infer_ds.batch(net_config.batch_size)
    
    # infer_ds = infer_ds.batch(26)

    print("inferring...")
    # infer and aggregate results
    aggregator = InferenceAggregator(top_k=1)
    for user, item, relation1, entity, relation2, hist_item, rating in infer_ds:
        del rating
        # print("inferring1...")
        result = network(item, relation1, entity, relation2, hist_item)
        # print("inferring4...")
        item_score = result[0]
        path_importance = result[1]
        aggregator.aggregate(user, item, relation1, entity, relation2, hist_item, item_score, path_importance)
        # print("inferring2...")
    # show recommendations with explanations
    explainer = steam.TextExplainer(translate_path)
    recomms = aggregator.recommend()
    for user, recomm in recomms.items():
        for item_rec in recomm.item_records:

            item_name = explainer.translate_item(item_rec.item)
            print(f"Recommend <{item_name}> to user:{user}, because:")
            string = "This is the flask app deployement of recommendation visualization.\n"
            string += f"<pre>Recommend <{item_name}> to user:{user}, because:\n"

            # show explanations
            explanation = 0
            for path in item_rec.paths:
                print(" - " + explainer.explain(path))
                string += " - " + explainer.explain(path) + "\n"
                explanation += 1
                if explanation >= 3:
                    break
            print("")
            string += "</pre>"
            return string

    # try:
    #     mox.file.copy_parallel(args.result_url, obs_result_url)
    #     print("Successfully Upload {} to {}".format(args.result_url, obs_result_url))
    # except Exception as e:
    #     print('moxing upload {} to {} failed: '.format(args.result_url, obs_result_url) + str(e))

if __name__ == '__main__':
    eval_tbnet()
