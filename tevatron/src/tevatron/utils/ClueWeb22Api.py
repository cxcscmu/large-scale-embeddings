import os
import gzip
import fastzipfile
import zipfile


class MARCOWebClueWeb22Api:
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
            

class ClueWeb22Api:

    def __init__(self, cw22id, cw22root_path):
        self.cw22id = cw22id
        self.cw22root_path = cw22root_path

    def get_base_filename_by_id(self, cw22id, cw22root_path, file_type='html'):
        html_path = self.cw22root_path + os.sep + file_type
        id_parts = cw22id.split('-')
        doc = int(id_parts[len(id_parts) - 1])

        language = id_parts[1][:2]
        segment = id_parts[1][:4]
        directory = id_parts[1]
        base_path = html_path + os.sep + language + os.sep + segment + os.sep + directory + os.sep
        base_filename = base_path + id_parts[1] + '-' + id_parts[2]
        return base_filename


    def get_html_from_warc(self):
        cw22id = self.cw22id
        cw22root_path = self.cw22root_path
        base_filename = self.get_base_filename_by_id(cw22id, cw22root_path)

        warc_path = base_filename + '.warc.gz'
        offset_path = base_filename + '.warc.offset'

        id_parts = cw22id.split('-')
        doc = int(id_parts[len(id_parts) - 1])

        #Get html from warc using offset
        offset_length = len('{:010d}\n'.format(0, 0))
        with open (warc_path,'rb') as f_warc:
            with open (offset_path, 'r') as f_offset:
                f_offset.seek(int(doc) * int(offset_length))
                start_bytes = int (f_offset.read (offset_length).strip())
                end_bytes =   int (f_offset.read (offset_length).strip())
                f_warc.seek(start_bytes)
                record = f_warc.read(end_bytes - start_bytes)
                record = gzip.decompress(record).decode('utf-8')

                #Remove the WARC header to get the htmlStr
                warc_header = ''
                for line in record.splitlines():
                    warc_header += line
                    warc_header += '\r\n'
                    if len(line.strip()) == 0:
                        break
                record = record[len(warc_header):]

                return record
   

    def get_json_record(self, record_type):
        cw22id = self.cw22id
        cw22root_path = self.cw22root_path
        base_filename = self.get_base_filename_by_id(cw22id, cw22root_path, file_type=record_type)

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


    def get_clean_text(self):
        record = self.get_json_record('txt')
        return record

    def get_inlinks(self):
        record = self.get_json_record('inlink')
        return record

    def get_outlinks(self):
        record = self.get_json_record('outlink')
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