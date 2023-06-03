import train
import predict


if __name__ == '__main__':
    model = train.MLP()
    trainer = train.trainer('RPT_FORECAST_STK_202305310911.csv', model)
    neg = []
    pos = []
    lexi_dict = {"大幅": 1, }

    # print(trainer._x_train)
    # trainer.train(10)
    # classifier = predict.predictor('咨询titles.txt', trainer)
    # classifier.debug()
    # classifer.run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
