from typing import Literal, Optional, Type, TypeVar
from pathlib import Path
from base64 import b64encode, b64decode
from pydantic import BaseModel


T = TypeVar('T', bytes, str)
ImageMimeType = Literal['image/jpeg', 'image/png', 'image/gif', 'image/webp']
IMAGE_MIME_TYPES = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']

def mime_type_from_path_or_url(path_or_url: str) -> ImageMimeType:
    ext = Path(path_or_url).suffix
    if ext == '.jpeg' or ext == '.jpg':
        return 'image/jpeg'
    if ext == '.png':
        return 'image/png'
    if ext == '.gif':
        return 'image/gif'
    if ext == '.webp':
        return 'image/webp'
    raise ValueError(f'Invalid image extension: {ext}')


class Image(BaseModel):    
    # data: bytes
    source: str|bytes
    format: Literal['base64', 'url', 'file'] = 'base64'
    mime_type: ImageMimeType = 'image/jpeg'
    

    def __init__(self,
                 source: bytes|str|Path,
                 format: Literal['base64', 'url', 'file'] = 'base64',
                 mime_type: ImageMimeType = 'image/jpeg',
                 cache_raw_bytes: bool = False,
                 cache_base64_bytes: bool = False,
                 cache_base64_string: bool = False
                 ):
        
        if isinstance(source, Path):
            source = str(source)
        if mime_type.endswith('jpg'):
            mime_type = 'image/jpeg'
        if not mime_type.startswith('image/'):
            mime_type = f'image/{mime_type}'

        BaseModel.__init__(self, source=source, format=format, mime_type=mime_type)
        self._raw_bytes = None
        self._base64_bytes = None
        self._base64_string = None
        self._cache_raw_bytes = cache_raw_bytes
        self._cache_base64_bytes = cache_base64_bytes
        self._cache_base64_string = cache_base64_string


    @classmethod
    def from_base64(cls, 
                    base64_data: str|bytes,
                    mime_type: ImageMimeType = 'image/jpeg'
                    ) -> 'Image':
        return cls(source=base64_data, format='base64', mime_type=mime_type)

    @classmethod
    def from_data_uri(cls, 
                      data_uri: str,
                      mime_type: Optional[ImageMimeType] = None
                      ) -> 'Image':
         
        if not data_uri.startswith('data:image/'):
            raise ValueError('Invalid data URI. ')
        
        split_uri = data_uri.split(';')        
        mime_type = mime_type or split_uri[0][5:] # strip 'data:'
        format, data = split_uri[-1].split(',', 1)        
        
        if not data:
            raise ValueError('No data in data URI.')
    
        return cls(source=data, format=format, mime_type=mime_type)
    

    @classmethod
    def from_url(cls, 
                 url: str,
                 mime_type: Optional[ImageMimeType] = None,
                 ) -> 'Image':
        
        mime_type = mime_type or mime_type_from_path_or_url(url)
        return cls(source=url, format='url', mime_type=mime_type)
    
    @classmethod
    def from_file(cls,
                      path: str|Path,
                      mime_type: Optional[ImageMimeType] = None,
                      ) -> 'Image':
        mime_type = mime_type or mime_type_from_path_or_url(path)
        return cls(source=path, format='file', mime_type=mime_type)


    @property
    def raw_bytes(self) -> bytes:
        if self._raw_bytes is not None:
            return self._raw_bytes
        
        raw_bytes = None
        if self.format == 'base64':
            raw_bytes = b64decode(self.base64_bytes)
            # return self._raw_bytes
        
        if self.format == 'url':
            raw_bytes = b'' # TODO: Get image from URL
            # return self._raw_bytes
        
        if self.format == 'file':
            with open(self.source, 'rb') as f:
                raw_bytes = f.read()
            # return self._raw_bytes
        
        if raw_bytes is None:
            raise ValueError(f'Invalid image format: {self.format}')

        if self._cache_raw_bytes:
            self._raw_bytes = raw_bytes

        return raw_bytes
        

    @property
    def base64_bytes(self) -> bytes:
        if self._base64_bytes is not None:
            return self._base64_bytes
        
        base64_bytes = None
        if self.format == 'base64':
            if isinstance(self.source, str):
                base64_bytes = self.source.encode('utf-8')
            elif isinstance(self.source, memoryview):        
                base64_bytes = self.source.tobytes()                
            else:
                base64_bytes = self.source                
                                                            
        if self.format == 'url' or self.format == 'file':
            base64_bytes = b64encode(self.raw_bytes)

        if base64_bytes is None:
            raise ValueError(f'Invalid image format: {self.format}')
        
        if self._cache_base64_bytes:
            self._base64_bytes = base64_bytes
        
        return base64_bytes


    @property
    def base64_string(self) -> str:
        if self._base64_string is not None:
            return self._base64_string
        
        if self.format == 'base64' and isinstance(self.source, str):
            base64_string = self.source            
        else:
            base64_string = self.base64_bytes.decode('utf-8')
        
        if self._cache_base64_string:
            self._base64_string = base64_string
            
        return base64_string
    

    @property
    def data_uri(self) -> str:
        return f'data:{self.mime_type};base64,{self.base64_string}'
        

    def __str__(self):
        return self.data_uri
    
    
    @property
    def source_string(self) -> str:
        if isinstance(self.source, str):
            return self.source
        if isinstance(self.source, memoryview):
            return self.source.tobytes().decode('utf-8')
                
        return self.source.decode('utf-8')
    

    @property
    def url(self) -> Optional[str]:
        if self.format != 'url':
            return None        
        return self.source_string
        
    @property
    def path(self) -> Optional[Path]:
        if self.format != 'file':
            return None        
        return Path(self.source_string)
    
