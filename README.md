# Neural Networks Modules

I implement from scratch and store in this repository neural networks modules such as convolution, correlation, attention and others. It serves me and other people educational purpose.
Each module is provided with tests that take output of pytorch modules as ground truth.

# Convolution

Convolution of *f* and *g* is denoted as *f* ⁕ *g* and defined as the sum of products *f* and *g* where one of them is reflected about y-axis and shifted:

![conv_image1](https://cdn-images-1.medium.com/max/800/1*9Ktq8rrD5iYhodp7cmn1pg.jpeg)

In general case:

![conv_image2](https://cdn-images-1.medium.com/max/800/1*FolIfDvMbAOS6Yp1YLztwA.png)

![conv_diag](https://cdn-images-1.medium.com/max/800/1*VucpQ8rpTlX0W45ThzzHKQ.png)

# Correlation

Correlation of *f* and *g* is denoted as *f*⋆*g* and defined as the sum of products *f* and *g* where one of them is shifted:

![corr_image1](https://cdn-images-1.medium.com/max/800/1*kNHYRsuXhiIJYhqgyTTRXw.jpeg)

In general case:

![corr_image2](https://cdn-images-1.medium.com/max/800/1*K93V7i99Vd1I0R2yeNeKFQ.png)

![corr_diag](https://cdn-images-1.medium.com/max/800/1*vPvaiTOObG5C_NtevFXH-w.png)
