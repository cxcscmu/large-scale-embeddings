import os
import gzip


class ClueWeb22Api:
    """
    ClueWeb22Api adapted from source: https://github.com/lemurproject/ClueWeb22/blob/main/ClueWeb22Api.py
    This version adpated to the specific file layout for the processed MARCOWeb corpus: e.g. MARCO_Web/train-2/de/de-00.json.gz
    """

    def __init__(self, cw22id, cw22root_path):
        self.cw22id = cw22id
        self.cw22root_path = cw22root_path

    def get_marcoweb_clean_text(self):
            record = self.get_marcoweb_json_record('txt')
            return record
    
    def get_base_filename(self, cw22id):
        html_path = self.cw22root_path + os.sep 
        id_parts = cw22id.split('-') # "clueweb22-en-{jjsongz_id}-{ddoc_id}"
        language = id_parts[1]
        base_path = html_path + os.sep + language + os.sep 
        base_filename = base_path + id_parts[1] + '-' + id_parts[2]
        return base_filename
        
    def get_marcoweb_json_record(self, record_type="txt"):
        cw22id = self.cw22id 
        # custom file path 
        base_filename = self.get_base_filename(cw22id)
        json_path = base_filename + '.json.gz'
        offset_path = base_filename + '.offset'

        id_parts = cw22id.split('-')
        doc = int(id_parts[len(id_parts) - 1])

        offset_length = len('{:010d}\n'.format(0, 0))
        with open (json_path,'rb') as f_json:
            with open (offset_path, 'r') as f_offset:
                f_offset.seek(int(doc) * int(offset_length))
                start_bytes = int (f_offset.read (offset_length).strip())
                end_bytes =   int (f_offset.read (offset_length).strip())
                f_json.seek(start_bytes)
                record = f_json.read(end_bytes - start_bytes)
                record = gzip.decompress(record).decode('utf-8')
                return record

# Create shard for the specific dataset 
def create_shards(data, num_shards, index):
    if not isinstance(data, list):
        raise ValueError("Input data must be a list.")

    # Compute shard size
    shard_size = len(data) // num_shards
    
    start_index = index * shard_size
    if index == num_shards - 1: 
        # final shard handle the residuals 
        end_index = len(data)
    else: 
        end_index = start_index + shard_size

    return data[start_index:end_index]