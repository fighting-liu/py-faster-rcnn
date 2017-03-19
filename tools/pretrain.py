from arg_parser import config, init_mysql
import pprint
import json

def bbox2seg(bb):
    x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
    return [[x1, y1, x1, y2, x2, y2, x2, y1]]

class AnnoData(object):
    def __init__(self, **kwargs):
        self.mysql = init_mysql(section_name=kwargs.get('mysql_conf_sec'), conf_file=kwargs.get('conf_file', 'pretrain.cfg'))

    def get_anno(self):
        bbox_sql = '''select
            b.id as image_id,
            a.source_id as source_id,
            a.id as id, FLOOR(a.x * b.width) as x, FLOOR(a.y * b.height) as y,
            CEIL(a.width * b.width) as width, CEIL(a.height * b.height) as height,
            0 as iscrowd,
            CEIL(a.width * b.width) * CEIL(a.height * b.height) as area,
            1 as category_id
        from anno_image_bndbox a
        left join ImageCacheXHS b
        on a.source_id = b.img and a.source = 1
        where a.source = 1 and a.status = 1
        limit 5000'''
        res = self.mysql.select(bbox_sql)

        for r in res:
            r['bbox'] = [r.pop('x'), r.pop('y'), r.pop('width'), r.pop('height')]
            if 'segmentation' not in r:
                r['segmentation'] = bbox2seg(r['bbox'])

        img_sql = '''select
            b.id as id, b.url as img_url, b.path as file_name, b.width as width, b.height as height, b.img
        from
            ImageCacheXHS b
        where
            b.img in
        (
            %s
        )
        ''' % (','.join(["%s"] * len(res)))
        img_res = self.mysql.select(img_sql, args=[x['source_id'] for x in res])

        anno = {
            'type':'instances',
            'categories':[{'supercategory':'none', 'id':1, 'name':'handbag'}],
            'images':img_res,
            'annotations':res
        }

        print json.dumps(anno)

    def get_class(self):
        sql = '''
            SELECT a.img, a.path, b.cat2 FROM ImageCache a INNER JOIN Htable b ON a.itemid = b.itemid;

        '''
        try:
            train_file = open(sys.argv[-2], 'w')
            val_file = open(sys.argv[-1], 'w')
        except:
            print >> sys.stderr, 'python %s train_file val_file' % (sys.argv[0])
            return
        res = self.mysql.select(sql)
        def count(x, y):
            x[y.get('cat2')] = 1 + x.get(y['cat2'], 0)
            return x
        cat_num = reduce(count, res, {})
        out_num = {}
        print cat_num

        for r in res:
          try:
            c2 = r['cat2']
            if out_num.get(c2, 0) + 1 > cat_num[c2] * 0.8:
                print >> val_file, r['path'].replace('/data/huebloom/img/H/', ''), c2[1:]
                continue
            else:
                out_num[c2] = out_num.get(c2, 0) + 1
                print >> train_file, r['path'].replace('/data/huebloom/img/H/', ''), c2[1:]
          except:
            print r, 'crashes'
            raise

    def visualize(self):
        with open(sys.argv[-1], 'r') as f:
            for line in f:
                pp = eval(line)
                sql = 'INSERT INTO anno_image_bndbox (%s) VALUES (%s)' % (','.join(pp.keys()), ','.join(['%s']*len(pp)))
                self.mysql.insert(sql, args=pp.values())

    def add_img(self):
        with open(sys.argv[-1], 'r') as f:
            ann = json.loads(f.read())
        ids = [x['id'] for x in ann['images']]

        res = self.mysql.select('select id, img from ImageCacheXHS where id in (%s)' % ','.join(["%s"] * len(ids)), args = ids)
        id_img_map = dict([(int(x['id']), x['img']) for x in res])

        for i in ann['images']:
            i['img'] = id_img_map[i['id']]

        print json.dumps(ann)

    def visualize_match(self):
        i = 0
        with open(sys.argv[-1], 'r') as f:
            for line in f:
                if i>=10:
                    break
                m = eval(line)
                for pp in m:
#                    pp.pop('score')
                    sql = 'INSERT INTO image_match (%s) VALUES (%s)' % (','.join(pp.keys()), ','.join(['%s']*len(pp)))
                    print sql, pp.values()
                    self.mysql.insert(sql, args=pp.values())

                i += 1

    def get_tops(self):
        i = 0
        with open(sys.argv[-1], 'r') as f:
            ids = f.read().split()
        print ids[:10]
        res = self.mysql.select('SELECT count(1) FROM image_match WHERE source = 1 AND source_id IN (%s)' % ','.join(['%s'] * len(ids)), args=ids)
        self.mysql.update('UPDATE ImageCacheXHS SET host="1" WHERE img IN (%s)' % ','.join(['%s'] * len(ids)), args=ids)
        print res

    def add_brand(self):
        def uniql(a):
            def dr(t, x):
                if x not in t:
                    t.append(x)
                return t
            a = reduce(dr, a, [])
            return a
        i = 0
        with open(sys.argv[-1], 'r') as f:
            for line in f:
                if i >= 10:
                    break
                x = line.split()
                print x
                if i == 0:
                    i += 1
                    continue
                b = filter(lambda m: m != 'Null', x)
                print uniql(reduce(lambda m, n: m + n.split('```'), b[1:], []))

                sql = u'INSERT INTO image_brand_style (img_id, brandname, created_at) SELECT img, %s, UNIX_TIMESTAMP() FROM ImageCacheXHS where itemid=%s' 
                self.mysql.insert(sql, args = [json.dumps(b[1:]), b[0]])
                i += 1

if __name__ == '__main__':
    import sys
    ad = AnnoData(mysql_conf_sec='mysql_dev')
    ad.add_brand()
