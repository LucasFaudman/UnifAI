from typing import Literal
from pydantic import BaseModel

ImageMediaType = Literal['image/jpeg', 'image/png', 'image/gif', 'image/webp']

class Image(BaseModel):    
    # data: bytes
    source: str|bytes
    format: Literal['base64', 'url', 'file'] = 'base64'
    media_type: ImageMediaType = 'image/jpeg'
    

    def __init__(self,
                 source: str|bytes,
                 format: Literal['base64', 'url', 'file'] = 'base64',
                 media_type: ImageMediaType = 'image/jpeg'
                 ):
        self.source = source
        self.format = format
        self.media_type = media_type
        self._data = None
        BaseModel.__init__(self, source=source, format=format, media_type=media_type)

    @property
    def data(self):
        if self._data:
            return self._data
        
        if self.format == 'base64':
            if isinstance(self.source, str):
                self._data = self.source.encode('utf-8')
            else:
                self._data = self.source
                if isinstance(self._data, memoryview):        
                    self._data = self._data.tobytes()
                                                
        elif self.format == 'url':
            self._data = b'' # TODO: Get image from URL
        
        elif self.format == 'file':
            with open(self.source, 'rb') as f:
                self._data = f.read()

        return self._data

        
    
class ImageFromBase64(Image):
    format: Literal['base64'] = 'base64'
    
    def __init__(self, 
                 base64_data: str|bytes,
                 media_type: ImageMediaType = 'image/jpeg'
                 ):
        super().__init__(source=base64_data, format='base64', media_type=media_type)
        


class ImageFromUrl(Image):
    format: Literal['url'] = 'url'

    def __init__(self, 
                 url: str,
                 media_type: ImageMediaType = 'image/jpeg',
                 read_on_init: bool = False
                 ):
        
        super().__init__(source=url, format='url', media_type=media_type)
        if read_on_init:
            assert self.data


class ImageFromFile(Image):
    format: Literal['file'] = 'file'

    def __init__(self,
                 filepath: str,
                 media_type: ImageMediaType = 'image/jpeg',
                 read_on_init: bool = False
                ):
            
            super().__init__(source=filepath, format='file', media_type=media_type)
            if read_on_init:
                assert self.data


# Image(source='data:image/jpeg;base64,abc', format='base64', media_type='image/jpeg')
# Image(source=b'abc', format='base64', media_type='image/jpeg')
# Image(source='https://example.com/image.jpg', format='url', media_type='image/jpeg')
# Image(source='image.jpg', format='file', media_type='image/jpeg')

# ImageFromBase64(base64_data='abc', media_type='image/jpeg')
# ImageFromUrl(url='https://example.com/image.jpg', media_type='image/jpeg')
# ImageFromFile(filepath='image.jpg', media_type='image/jpeg')