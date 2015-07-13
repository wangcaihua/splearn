from distutils.core import setup


setup (# Distribution meta-data
	name = 'splearn', 
	version = '0.3', 
	description = "sparse learning package for python",
	author = 'Caihua Wang',
	author_email = '490419716@qq.com',
	packages = ['splearn', 'splearn/loss', 'splearn/proreg', 'splearn/solver', 'splearn/linmod', 'splearn/matdec', 'splearn/test'],
	package_dir = {'splearn': 'splearn'}, 
	package_data = {'splearn':['doc/splearn_document.pdf', 'test/dataset/clasdata', 'test/dataset/regudata']}
	)
