## EDA data penjualan mobil menggunakan Orange
Dataset dapat ditemukan pada link berikut: https://www.kaggle.com/CooperUnion/cardataset/notebooks <br><br>
Dari yang sudah kami analisis menggunakan Jupyter Notebook, kami menemukan bahwa terdapat banyak outlier pada variabel Popularity. <br>
![](https://user-images.githubusercontent.com/57984390/95144651-6033c000-07a3-11eb-9108-345b4d14c8a3.PNG) <br>
Dengan menggunakan Box Plot, kita dapat melihat bahwa ada data dengan nilai Popularity sangat jauh dari mean, dan hampir mendekati 6000. <br>
![](https://user-images.githubusercontent.com/57984390/95144655-645fdd80-07a3-11eb-8074-504d54c67ee6.PNG) <br>
Setelah kami amati pada Data Table, ternyata outlier cukup banyak. Setelah mencari beberapa nama model, kami berhipotesis bahwa mobil-mobil ini populer karena merupakan tipe-tipe mobil keluarga yang ekonomis. <br>
![](https://user-images.githubusercontent.com/57984390/95144659-66c23780-07a3-11eb-9043-3b3a512717b9.PNG) <br>
Dengan menggunakan scatter plot, dapat kita amati bahwa mobil-mobil yang memiliki nilai Popularity di atas normal rata-rata memiliki harga retail yang cukup rendah. Selain itu, dapat kita amati bahwa ada beberapa mobil yang memiliki harga retail yang sangat tinggi. <br>
![](https://user-images.githubusercontent.com/57984390/95144666-69bd2800-07a3-11eb-8a3b-ba7e4c245904.PNG)
![](https://user-images.githubusercontent.com/57984390/95145799-65ded500-07a6-11eb-82b9-2f0745478ecb.PNG) <br>
Rupanya mobil-mobil tersebut adalah mobil-mobil mewah dengan tenaga kuda yang tinggi serta silinder yang banyak.
