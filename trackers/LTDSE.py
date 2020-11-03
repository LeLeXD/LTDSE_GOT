from __future__ import absolute_import

from . import Tracker


class LTDSE(Tracker):

    def __init__(self):
        super(LTDSE, self).__init__(
            name='LTDSE',
            is_deterministic=True)

    def init(self, image, box):
        self.box = box

    def update(self, image):
        return self.box