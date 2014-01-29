require 'fileutils'

KAGGLE_TRAIN = File.join(ENV['HOME'], "data/mnist-kaggle/train.csv")
KAGGLE_TEST = File.join(ENV['HOME'], "data/mnist-kaggle/test.csv")
TRAIN_DATA = "./data/train_data.mat"
TRAIN_DATA32 = "./data/train_data32.mat"
TRAIN_LABEL = "./data/train_labels.mat"
TEST_DATA = "./data/test_data.mat"
TEST_DATA32 = "./data/test_data32.mat"
COLUMNS = 28

def make_image(padding, pixels)
  image = pixels.each_slice(COLUMNS).map do |column|
    padding.times do
      column.unshift(0).push(0)
    end
    column
  end
  padding.times do
    pad = Array.new(COLUMNS+padding*2, 0)
    image.unshift(pad).push(pad)
  end
  image
end

def make_train_file(padding, train_data_file, train_label_file)
  header = false
  data = []
  labels = []
  File.read(KAGGLE_TRAIN).split("\n").each do |line|
    if !header
      header = true
      next
    end
    label, *pixels = line.split(",").map{|v| v.to_i}
    image = make_image(padding, pixels)
    data << image.flatten
    labels << label
  end
  File.open(train_data_file, "w") do |f|
    f.puts("1 #{data.size} #{data[0].size} 1 #{data[0].size}")
    data.each do |vec|
      f.puts vec.flatten.map{|v| sprintf("%E", v)}.join(" ")
    end
  end
  File.open(train_label_file, "w") do |f|
    f.puts("1 #{labels.size} 1 1 1")
    labels.each do |label|
      f.puts sprintf("%E", label)
    end
  end
end

def make_test_file(padding, test_data_file)
  header = false
  data = []
  File.read(KAGGLE_TEST).split("\n").each do |line|
    if !header
      header = true
      next
    end
    pixels = line.split(",").map{|v| v.to_i}
    image = make_image(padding, pixels)
    data << image.flatten
  end
  File.open(test_data_file, "w") do |f|
    f.puts("1 #{data.size} #{data[0].size} 1 #{data[0].size}")
    data.each do |vec|
      f.puts vec.flatten.map{|v| sprintf("%E", v)}.join(" ")
    end
  end
end

FileUtils.mkdir_p(File.dirname(TRAIN_DATA))

# make raw data
make_test_file(0, "data_tmp")
puts `nv_matrix_text2bin data_tmp #{TEST_DATA}`
puts `rm data_tmp`
make_train_file(0, "data_tmp", "label_tmp")
puts `nv_matrix_text2bin data_tmp #{TRAIN_DATA}`
puts `nv_matrix_text2bin label_tmp #{TRAIN_LABEL}`
puts `rm data_tmp`
puts `rm label_tmp`

# make padding-2px data
make_test_file(2, "data_tmp")
puts `nv_matrix_text2bin data_tmp #{TEST_DATA32}`
puts `rm data_tmp`
make_train_file(2, "data_tmp", "label_tmp")
puts `nv_matrix_text2bin data_tmp #{TRAIN_DATA32}`
puts `rm data_tmp`
puts `rm label_tmp`
