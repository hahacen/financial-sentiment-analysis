import train
import predict


if __name__ == '__main__':
    model = train.MLP(4, 3)
    trainer = train.trainer('train.csv', model)
    classifier = predict.predictor('咨询titles.txt', trainer)
    classifier.debug()
    # classifer.run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
