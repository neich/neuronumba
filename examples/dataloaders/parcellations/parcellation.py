class Parcellation:
    def get_coords(self):
        raise NotImplemented('Should have been implemented by subclass!')

    def get_region_labels(self):
        raise NotImplemented('Should have been implemented by subclass!')

    def get_region_short_labels(self):
        raise NotImplemented('Should have been implemented by subclass!')

    def get_cortices(self):
        raise NotImplemented('Should have been implemented by subclass!')

    def get_RSN(self, useLR=False):
        raise NotImplemented('Should have been implemented by subclass!')

    def get_data(self, attribute, extra=None):
        if attribute == 'coords':
            return self.get_coords()
        elif attribute == 'labels':
            return self.get_region_labels()
        elif attribute == 'short_labels':
            return self.get_region_short_labels()
        elif attribute == 'cortices':
            return self.get_cortices()
        elif attribute == 'RSN':
            return self.get_RSN()
        else:
            raise NotImplemented('Should have been implemented by subclass!')
