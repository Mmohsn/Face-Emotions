class DataPreprocessing:
    def __init__(self, data):
        self.data = pd.read_csv(str(data))
        self.data_labels = {0: 'Angry', 1: 'Disgust', 2: 'Afraid', 3: 'Ecstatic', 4: 'NO REACTION!', 5: 'Unhappy', 6: 'Surprised'}
        self.new_data = self.data.copy()
        self.new_data['Label'] = self.new_data['emotion'].map(self.data_labels)
        self.classes = to_categorical(self.data[['emotion']], num_classes=7)
        self.data.dropna(inplace=True)
        self.pixel_list = self.new_data['pixels'].to_list()

    def preprocess_and_split_data(self):
        Arr = []
        for pix in self.pixel_list:
            img = [int(pixel) for pixel in pix.split(' ')]
            img = np.asarray(img).reshape(48, 48)
            Arr.append(img.astype('float32') / 255.0)

        X = np.array(Arr)
        X = np.expand_dims(X, -1)
        print(X.shape)
        y = self.new_data['emotion'].values
        print(y.shape)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        return x_train, x_test, y_train, y_test