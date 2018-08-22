# Online Handwritten Character Recognition with capacitive sensors

In this project I evaluated different machine learning models on the task of online handwritten character recognition. 

The dataset contains samples for six different letters (P, E, A, W, S and B), which can be written as capital, lower case or cursive letter, and a noise class with invalid symbols and other, not to be classified letters.
The samples originate from (x,y) coordinate pairs taken with the Low Power Projected Capacitive Touchpad Development Kit by Microchip [1].
The dataset was collected with the capture_data script and contains 2 different writers, each contributing 350 of the total 700 samples.
The use of the public UNIPEN dataset for training the classifiers was also explored, however it was not possible to generalize from the UNIPEN data to the data gathered with the Toolkit.

The following steps of data preparation were conducted:
- Feature engineering: 10 features were extracted to boost the performance of the models. The features are oriented on a paper by Schmidhuber et al. [2].
- Data normalization: scale each of the input dimensions to zero mean and unit variance
- Data augmentation: Due to the limited dataset, the training data is artificially augmented by random shift, up/down sample, reverse, clinch/stretch and rotate. For each entry in the training set, 4 augmented samples are created.

Additionally, dimensionality reduction with PCA was evaluated but led to lower accuracy and was therefore discarded.

The train/validation/test split is 80% / 5% / 15%. Together with the data augmentation, we have 2800 training samples, 35 validation samples and 105 test samples.

Results
-
As classification models I evaluated Recurrent Neural Networks (RNN) with Gated Recurrent Units (GRU), Convolutional Neural Networks and other, more classical machine learning models like Hidden Markov Models (HMM), Random Forests, Multi-Layer Perceptrons (MLP) and Support Vector Machines (SVM). However, the classical models did not provide a satisfying accuracy.

Model | Accuracy
----- | --------
RNN | 91.4%
CNN | 86.7%
MLP | 80.0%
HMM | 79.1%
Linear SVM | 74.3%
RandomForest | 72.4%
Kernel SVM | 71.4%

Since the Recurrent Neural Networks gave the most promising results, I tested several recent approaches to improve their performance like different attention mechanisms [3, 4], layer normalization [5] and Stochastic Weight Averaging [6].

Model | Accuracy
----- | --------
baseline RNN + SWA | 95.2%
baseline RNN | 91.4%
baseline RNN + layerNorm | 91.4%
attention [3] RNN | 91.4%
attention [4] RNN | 89.5%
attention [3] RNN + layerNorm | 89.5%
attention [4] RNN + layerNorm | 88.5%

Conclusively, one can see that the findings of [6] can be confirmed by this work, stochastic weight averaging boosts the performance of NNs. The attention mechanisms however, which led to breakthroughs in e.g. Neural Machine Translation, turn out to be not as helpful for this task. In addition, also LayerNorm did not lead to a higher accuracy.

Furthermore, it is worth noting that the winning architecture had only 5671 parameters equating to 22.15 kB in 32-bit floating point numbers. Hence, this classifier could be used to create a low cost device (e.g. smart home switch) controlled by handwritten character recognition.


References
-
[1] Low Power Projected Capacitive Touchpad Development Kit, https://www.microchip.com/DevelopmentTools/ProductDetails/dm160219.

[2] Liwicki, M., Graves, A., Fernàndez, S., Bunke, H., & Schmidhuber, J. (2007). A novel approach to on-line handwriting recognition based on bidirectional long short-term memory networks. In Proceedings of the 9th International Conference on Document Analysis and Recognition, ICDAR 2007.

[3] Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.

[4] Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016). Hierarchical attention networks for document classification. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (pp. 1480-1489).

[5] Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. arXiv preprint arXiv:1607.06450.

[6] Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., & Wilson, A. G. (2018). Averaging Weights Leads to Wider Optima and Better Generalization. arXiv preprint arXiv:1803.05407.

License
-
MIT License

Copyright (c) [2018] [Tobias Fischer]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
