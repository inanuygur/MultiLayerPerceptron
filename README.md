# MultiLayerPerceptron
## An implementation of Multi Layer Perceptron for MNIST dataset

This project is an basic implementation of multi layer perceptron without using any ML libraries.
The base idea goes parallel to Simon Haykin's Neural Networks: A Comprehensice Foundation book - chapter 4 - Multilayer Perceptrons.
Proposed network learns with back propagation and uses only sigmoid activation function. 

Different learning rates and proper momentum rates are used for each layer. 
Each layer's weighth initiated within the range -1 and 1 with 0.1 margin. 

## How to Prepare The Dataset
Download the MNIST dataset from "The MNIST Database" site using 'http://yann.lecun.com/exdb/mnist/' link. Move the files to directory that "prepdataset.py" code is in and run the code. It will give numpy stacks that will be used in the main code. 

## How to Train
Move "multilayerPerceptron.py" code next to prepdataset.py and run multilayerPerceptron.py. Process can and will be slow due to inefficiency in the code but result will be display on the screen as one iteration completes. 
Default stopping point for training session is an loss error value. Once the error goes under given value, training will stop. Obtained weights can be stored in a numpy array file. 


# MultiLayerPerceptron
## MNIST veri kümesi için Çok Katmanlı Algılayıcı'nın bir uygulaması

Bu proje, herhangi bir ML kitaplığı kullanmadan çok katmanlı algılayıcının temel bir uygulamasıdır.
Temel fikir, Simon Haykin'in Neural Networks: A Comprehensice Foundation kitabına paraleldir - Bölüm 4 - Çok Katmanlı Algılayıcılar.
Önerilen ağ, geri yayılımla öğrenir ve yalnızca sigmoid etkinleştirme işlevini kullanır.

Her katman için farklı öğrenme oranları ve uygun momentum oranları kullanılır.
Her katmanın ağırlığı 0,1 marj ile -1 ve 1 aralığında başlatılır.

## Veri Kümesi Nasıl Hazırlanır
MNIST veri kümesini "The MNIST Veritabanı" sitesinden 'http://yann.lecun.com/exdb/mnist/' bağlantısını kullanarak indirin. Dosyaları "prepdataset.py" kodunun bulunduğu dizine taşıyın ve kodu çalıştırın. Ana kodda kullanılacak uyuşmuş yığınlar verecektir.

## Nasıl eğitilir
"MultilayerPerceptron.py" kodunu prepdataset.py'nin yanına taşıyın ve multilayerPerceptron.py'yi çalıştırın. Koddaki verimsizlik nedeniyle işlem yavaş olabilir ve olacaktır, ancak bir yineleme tamamlandığında sonuç ekranda görüntülenecektir.
Egzersiz seansı için varsayılan durma noktası, bir kayıp hata değeridir. Hata verilen değerin altına düştüğünde eğitim duracaktır. Elde edilen ağırlıklar bir numpy dizi dosyasında saklanabilir.
