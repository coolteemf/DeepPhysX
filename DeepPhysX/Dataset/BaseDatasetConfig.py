class DatasetConfig:

    def __init__(self, dataset_class, partition_size, dataset_dir=None, generate_data=False, shuffle_dataset=False):
        # Dataset variables
        self.dataset_class = dataset_class
        self.maxSize = int(partition_size * 1e9)
        self.datasetConfig = self.maxSize
        # DatasetManager variables
        self.datasetDir = dataset_dir
        self.existingDataset = False if dataset_dir is None else True
        self.generateData = generate_data
        self.shuffleDataset = shuffle_dataset
        # Description
        self.descriptionName = "CORE DatasetConfig"
        self.description = ""

    def createDataset(self):
        return self.dataset_class(self.datasetConfig)

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\n{}\n".format(self.descriptionName)
            self.description += "   (dataset) Dataset class: {}\n".format(self.dataset_class.__name__)
            self.description += "   (dataset) Max size: {}\n".format(self.maxSize)
            self.description += "   (dataset) Dataset config: {}\n".format(self.datasetConfig)
            self.description += "   (datasetManager) Dataset dir: {}\n".format(self.datasetDir)
            self.description += "   (datasetManager) Existing dataset: {}\n".format(self.existingDataset)
            self.description += "   (datasetManager) Generate data: {}\n".format(self.generateData)
            self.description += "   (datasetManager) Shuffle dataset: {}\n".format(self.shuffleDataset)
        return self.description
