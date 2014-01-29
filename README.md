# Kaggle-MNIST

My code for Kaggle-MNIST competition.

## Build Requirements

- [libnv develop branch](https://github.com/nagadomi/nv/tree/develop)</a>

## Results

### raw_lmca/

<table>
  <tbody>
    <tr>
      <td>Preprocessing</td><td>none</td>
    </tr>
    <tr>
      <td>Feature extraction</td><td>raw data</td>
    </tr>
    <tr>
      <td>Postprocessing</td><td>L2 normalization, Large Margin Component Analysis</td>
    </tr>
    <tr>
      <td>Classifier</td><td>k-Nearest Neighbors</td>
    </tr>
    <tr>
      <td>Result</td><td>0.98100</td>
    </tr>
  </tbody>
</table>

### rectangle_whitend_mlp/

<table>
  <tbody>
    <tr>
      <td>Preprocessing</td><td>2px padding</td>
    </tr>
    <tr>
      <td>Feature extraction</td><td>Rectangle Features</td>
    </tr>
    <tr>
      <td>Postprocessing</td><td>Local standardization, ZCA Whitening, Global standardization</td>
    </tr>
    <tr>
      <td>Classifier</td><td>2-Layer NN (1024 hidden units, Softmax, Dropout & DropConnect)</td>
    </tr>
    <tr>
      <td>Result</td><td>0.98829</td>
    </tr>
  </tbody>
</table>
