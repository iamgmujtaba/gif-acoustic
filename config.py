import argparse

def parse_opts():

    parser = argparse.ArgumentParser()
    
    #Paths
    parser.add_argument('--save_dir', default='./output/', type=str, help='Where to save training outputs.')
    parser.add_argument("--video_path", type=str, default="./data/walter/", help="Path of the files that will downloaded from the server")
    
    # Dataset
    parser.add_argument('--dataset_path', default='./data/gtzan/', type=str, help='Path to location of dataset images')
    parser.add_argument('--dataset', default='gtzan', type=str, help='Dataset string (gtzan |)')
    parser.add_argument('--num_classes', default=10, type=int, help= 'Number of classes (gtzan: 10, )')

    # Models (general)
    parser.add_argument('--model', default='cnn', type=str, help='( inceptionV3 | efficientNet | resnext | densenet)')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')

    # Optimization
    parser.add_argument('--early_stopping_patience', default=10, type=int, help='Early stopping patience (number of epochs)')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch Size')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train for')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate (divided by 10 while training by lr-scheduler)')    
    
    #Misc
    parser.add_argument("--walter_ip", default='http://000.000.000.000/', type=str, help='Server IP address')

    return parser.parse_args()
