# -*- coding: utf-8 -*-
import bz2
import datetime
import gzip
import io
import lzma
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
import zlib

"""
Python中自带的不同压缩算法和打包技术的使用案例和对比

- https://docs.python.org/3/library/archiving.html
"""


def zlib_compression(src_file=None, tgt_file=None, new_src_file=None, is_stream=False):
    """
    zlib: https://docs.python.org/3/library/zlib.html

    There are known incompatibilities between the Python module and versions of the zlib library
    earlier than 1.1.3; 1.1.3 has a security vulnerability, so we recommend using 1.1.4 or later.
    """
    print(f"构建模块时所用的 zlib 库的版本: {zlib.ZLIB_VERSION}")
    print(f"解释器所加载的 zlib 库的版本: {zlib.ZLIB_RUNTIME_VERSION}")

    if src_file is None and tgt_file is None and new_src_file is None:
        print("使用zlib压缩和解压字符串")
        data = b"Lots of content here"
        comped_data = zlib.compress(data)
        decomped_data = zlib.decompress(comped_data)
        print(decomped_data)

        # Data compression ratio
        comp_ratio = len(data) / len(comped_data)
        print(f"压缩率为：{comp_ratio}")
        # Check equality to original object after round-trip
        print(f"解压后与原始内容一致：{decomped_data == data}")
    else:
        print("使用zlib压缩和解压文件")
        assert tgt_file.endswith('.zlib')
        if not is_stream:
            with open(src_file, 'rb') as src, open(tgt_file, 'wb') as tgt:
                # 参数 level 为压缩等级，是整数，可取值为 0 到 9 或 -1。
                # 1 (Z_BEST_SPEED) 表示最快速度和最低压缩率
                # 9 (Z_BEST_COMPRESSION) 表示最慢速度和最高压缩率
                # 0 (Z_NO_COMPRESSION) 表示不压缩。参数默认值为 -1 (Z_DEFAULT_COMPRESSION)。
                # Z_DEFAULT_COMPRESSION 是速度和压缩率之间的平衡 (一般相当于设压缩等级为 6)。
                data = src.read()
                tgt.write(zlib.compress(data, level=9))

                hash_code = zlib.adler32(data) & 0xffffffff
                print(hash_code)

            with open(tgt_file, 'rb') as src, open(new_src_file, 'wb') as tgt:
                data = zlib.decompress(src.read())
                tgt.write(data)

                hash_code = zlib.adler32(data) & 0xffffffff
                print(hash_code)
        else:
            with open(src_file, "rb") as src, open(tgt_file, "wb") as tgt:
                # 返回一个压缩对象，用来按照数据流的形式进行压缩，可以避免一次性占用太多的内存。
                comp_obj = zlib.compressobj(level=9)

                # https://github.com/rucio/rucio/blob/82a4fe7f3b12120c5815fc1f6b6d231f24949268/lib/rucio/common/utils.py#L284-L306
                hash_code = 1
                while data := src.read(1024):
                    hash_code = zlib.adler32(data, hash_code)

                    # comp_obj.compress: 压缩 data 并返回 bytes 对象，
                    # 这个对象含有 data 的部分或全部内容的已压缩数据。
                    # 所得的对象必须拼接在上一次调用 compress() 方法所得数据的后面。
                    # 缓冲区中可能留存部分输入以供下一次调用。
                    comped_data = comp_obj.compress(data)
                    tgt.write(comped_data)

                # comp_obj.flush: 压缩所有缓冲区的数据并返回已压缩的数据。
                comped_data = comp_obj.flush()
                tgt.write(comped_data)
                print(hash_code & 0xffffffff)

            with open(tgt_file, 'rb') as src, open(new_src_file, 'wb') as tgt:
                comp_obj = zlib.decompressobj()
                hash_code = 1
                while data := src.read(1024):
                    decomped_data = comp_obj.decompress(data)
                    tgt.write(decomped_data)

                    hash_code = zlib.adler32(decomped_data, hash_code)

                decomped_data = comp_obj.flush()
                tgt.write(decomped_data)

                hash_code = zlib.adler32(decomped_data, hash_code)
                print(hash_code & 0xffffffff)


def gzip_compression(src_file=None, tgt_file=None, new_src_file=None, use_copy=None,
                     is_stream=None):
    """
    gzip: https://docs.python.org/3/library/gzip.html

    此模块提供的简单接口帮助用户压缩和解压缩文件，功能类似于 GNU 应用程序 gzip 和 gunzip。

    数据压缩由 zlib 模块提供。
    gzip 模块提供 GzipFile 类和 open()、compress()、decompress() 几个便利的函数。
    GzipFile 类可以读写 gzip 格式的文件，还能自动压缩和解压缩数据，
    这让操作压缩文件如同操作普通的 file object 一样方便。
    """
    if src_file is None and tgt_file is None and new_src_file is None:
        print("使用gzip压缩和解压字符串")
        data = b"Lots of content here"
        comped_data = gzip.compress(data=data)
        decomped_data = gzip.decompress(data=comped_data)
        print(decomped_data)

        # Data compression ratio
        comp_ratio = len(data) / len(comped_data)
        print(f"压缩率为：{comp_ratio}")
        # Check equality to original object after round-trip
        print(f"解压后与原始内容一致：{decomped_data == data}")
    else:
        print("使用gzip压缩和解压文件")
        assert tgt_file.endswith('gz')
        with open(src_file, "rb") as src, gzip.open(tgt_file, "wb", compresslevel=9) as tgt:
            if not is_stream:
                if use_copy:
                    shutil.copyfileobj(src, tgt)
                else:
                    tgt.write(src.read())
            else:
                while data := src.read(1024):
                    tgt.write(data)
                tgt.flush()

        with gzip.open(tgt_file, "rb") as src, open(new_src_file, "wb") as tgt:
            if not is_stream:
                if use_copy:
                    shutil.copyfileobj(src, tgt)
                else:
                    tgt.write(src.read())
            else:
                while data := src.read(1024):
                    tgt.write(data)


def bzip2_compression(src_file=None, tgt_file=None, new_src_file=None, is_stream=False):
    """
    bz2: https://docs.python.org/3/library/bz2.html

    此模块提供了使用 bzip2 压缩算法压缩和解压数据的一套完整的接口。

    bz2 模块包含：

    - 用于读写压缩文件的 open() 函数和 BZ2File 类。
    - 用于增量压缩和解压的 BZ2Compressor 和 BZ2Decompressor 类。
    - 用于一次性压缩和解压的 compress() 和 decompress() 函数。
    """
    if src_file is None and tgt_file is None and new_src_file is None:
        print("使用bzip2算法压缩和解压字符串")
        data = b"Lots of content here"
        comped_data = bz2.compress(data)
        decomped_data = bz2.decompress(comped_data)
        print(decomped_data)

        # Data compression ratio
        comp_ratio = len(data) / len(comped_data)
        print(f"压缩率为：{comp_ratio}")
        # Check equality to original object after round-trip
        print(f"解压后与原始内容一致：{decomped_data == data}")
    else:
        print("使用bzip2算法压缩和解压文件")
        assert tgt_file.endswith('bz2')
        # bz2.open:
        # 对于二进制模式，这个函数等价于 BZ2File 构造器:
        #   BZ2File(filename, mode, compresslevel=compresslevel)。
        # 对于文本模式，将会创建一个 BZ2File 对象，并将它包装到一个 io.TextIOWrapper 实例中，
        #   此实例带有指定的编码格式、错误处理行为和行结束符。
        with open(src_file, "rb") as src, bz2.open(tgt_file, 'wb') as tgt:
            if not is_stream:
                tgt.write(src.read())
            else:
                while data := src.read(1024):
                    tgt.write(data)

        with bz2.open(tgt_file, 'rb') as src, open(new_src_file, 'wb') as tgt:
            if not is_stream:
                tgt.write(src.read())
            else:
                while data := src.read(1024):
                    tgt.write(data)


def lzma_compression(src_file=None, tgt_file=None, new_src_file=None, is_stream=False):
    """
    lzma: https://docs.python.org/3/library/lzma.html

    此模块提供了可以压缩和解压缩使用 LZMA 压缩算法的数据的类和便携函数。
    其中还包含支持 xz 工具所使用的 .xz 和旧式 .lzma 文件格式的文件接口，以及相应的原始压缩数据流。

    此模块所提供了接口与 bz2 模块的非常类似。
    请注意 LZMAFile 和 bz2.BZ2File 都 不是 线程安全的。
    因此如果你需要在多个线程中使用单个 LZMAFile 实例，则需要通过锁来保护它。
    """
    if src_file is None and tgt_file is None and new_src_file is None:
        print("使用lzma算法压缩和解压字符串")
        data = b"Lots of content here"
        comped_data = lzma.compress(data)
        # format 参数指定应当被使用的容器格式。
        # 默认值为 FORMAT_AUTO，它可以解压缩 .xz 和 .lzma 文件。
        # 其他可能的值为 FORMAT_XZ, FORMAT_ALONE 和 FORMAT_RAW。
        # https://docs.python.org/zh-cn/3/library/lzma.html#lzma.LZMACompressor
        decomped_data = lzma.decompress(comped_data, format=lzma.FORMAT_AUTO)
        print(decomped_data)

        # Data compression ratio
        comp_ratio = len(data) / len(comped_data)
        print(f"压缩率为：{comp_ratio}")
        # Check equality to original object after round-trip
        print(f"解压后与原始内容一致：{decomped_data == data}")
    else:
        print("使用lzma算法压缩和解压文件")
        assert tgt_file.endswith('.xz') or tgt_file.endswith('.lzma')
        # 除了更加 CPU 密集，使用更高的预设等级来压缩还需要更多的内存（并产生需要更多内存来解压缩的输出）。
        # 例如使用预设等级 9 时，一个 LZMACompressor 对象的开销可以高达 800 MiB。
        # 出于这样的原因，通常最好是保持使用默认预设等级。
        with open(src_file, 'rb') as src, lzma.open(tgt_file, 'wb',
                                                    preset=lzma.PRESET_DEFAULT) as tgt:
            if is_stream:
                tgt.write(src.read())
            else:
                while data := src.read(1024):
                    tgt.write(data)

        with lzma.open(tgt_file, 'rb') as src, open(new_src_file, 'wb') as tgt:
            if is_stream:
                tgt.write(src.read())
            else:
                while data := src.read(1024):
                    tgt.write(data)


def zip_compression(src_root, tgt_file, new_src_root1, new_src_root2, use_cmd):
    """
    zip: https://docs.python.org/3/library/zipfile.html

    ZIP 文件格式是一个常用的归档与压缩标准。这个模块提供了创建、读取、写入、添加及列出 ZIP 文件的工具。
    此模块目前不能处理分卷 ZIP 文件。可以处理使用 ZIP64 扩展（超过 4 GB 的 ZIP 文件）的 ZIP 文件。
    它支持解密 ZIP 归档中的加密文件，但是目前不能创建一个加密的文件。
    解密非常慢，因为它是使用原生 Python 而不是 C 实现的。
    """
    if not use_cmd:
        src_files = [os.path.join('./data/src/a', x) for x in os.listdir('./data/src/a')]
        # 如果 allowZip64 为 True (默认值) 则当 zipfile 大于 4 GiB 时 zipfile 将创建使用 ZIP64 扩展的ZIP文件。
        # 如果该参数为 false 则当 ZIP 文件需要 ZIP64 扩展时 zipfile 将引发异常.
        # compresslevel 形参控制在将文件写入归档时要使用的压缩等级。
        # - 当使用 ZIP_STORED 或 ZIP_LZMA 时无压缩效果。
        # - 当使用 ZIP_DEFLATED 时接受整数 0 至 9 (更多信息参见 zlib)。
        # - 当使用 ZIP_BZIP2 时接受整数 1 至 9 (更多信息参见 bz2)。
        with zipfile.ZipFile(tgt_file, 'w',
                             allowZip64=True,
                             compression=zipfile.ZIP_BZIP2,
                             compresslevel=9) as tgt:
            print(f"创建zip文件{tgt.filename}来打包{src_files}")
            # 如果创建文件时使用 'w', 'x' 或 'a' 模式并且未向归档添加任何文件就执行了 closed，
            # 则会将适当的空归档 ZIP 结构写入文件。
            for src_file in src_files:
                # 归档名称arcname应当是基于归档根目录的相对路径，也就是说，它们不应以路径分隔符开头。
                tgt.write(filename=src_file,
                          arcname=os.path.relpath(path=src_file,
                                                  start=os.path.dirname(
                                                      os.path.dirname(src_file))))

        with zipfile.ZipFile(tgt_file, 'r') as src:
            if broken_file := src.testzip() is not None:
                print(f"{tgt_file}中文件{broken_file}损坏")
            print(f"{tgt_file}中的文件：")
            src.printdir(file=sys.stdout)  # 也可以是具体的文件
            # 设置 pwd 为用于提取已加密文件的默认密码。
            src.setpassword(pwd=None)

            member_namelist = src.namelist()
            # ZipInfo 类的实例会通过 getinfo() 和 ZipFile 对象的 infolist() 方法返回。
            # 每个对象将存储关于 ZIP 归档的一个成员的信息。
            first_member_info = src.getinfo(name=member_namelist[0])
            member_infolist = src.infolist()
            print(f"第一个归档成员的信息是否一致：{first_member_info == member_infolist[0]}")

            print(member_namelist[0])
            # mode should be 'r' to read a file already in the ZIP file,
            # or 'w' to write to a file newly added to the archive.
            with src.open(member_namelist[0], 'r') as tgt:
                print(tgt.read().decode('utf-8')[:5])

            for member_info in member_infolist:
                print(f"如果此归档成员是一个目录则返回 True, 目录应当总是以 / 结尾: {member_info.is_dir()}")
                print(f"归档中的文件名称: {member_info.filename}")
                print(f"上次修改存档成员的时间和日期: {member_info.is_dir()}")
                print(f"已压缩数据的大小: {member_info.compress_size}")
                print(f"未压缩文件的大小: {member_info.file_size}")
                # 如果一个成员文件名为绝对路径，则将去掉驱动器/UNC共享点和前导的（反）斜杠，
                # 例如: ///foo/bar 在 Unix 上将变为 foo/bar，
                # 而 C:\foo\bar 在 Windows 上将变为 foo\bar。
                # 并且一个成员文件名中的所有 ".." 都将被移除，
                # 例如: ../../foo../../ba..r 将变为 foo../ba..r。
                # 在 Windows 上非法字符 (:, <, >, |, ", ?, and *) 会被替换为下划线 (_)。
                src.extract(member=member_info, path=new_src_root1, pwd=None)  # pwd 是用于解密文件的密码。
            # 警告 绝不要未经预先检验就从不可靠的源中提取归档文件。 这样有可能在 path 之外创建文件，
            # 例如某些成员具有以 "/" 开始的文件名或带有两个点号 ".." 的文件名。
            # 此模块会尝试防止这种情况。
            # 参见 extract() 的注释。
            src.extractall(path=new_src_root2, members=member_infolist, pwd=None)

        with zipfile.ZipFile(tgt_file, 'a') as src:
            if broken_file := src.testzip() is not None:
                print(f"{tgt_file}中文件{broken_file}损坏")
            print(f"输出文件内容目录：{src.printdir()}")
            first_member_info = src.infolist()[0]
            print(first_member_info.filename)

            with src.open('c/' + first_member_info.filename, 'w', force_zip64=True) as tgt:
                # 这一操作也是用于添加文档到存档中
                tgt.write(
                    f"{datetime.datetime.now()}写入到文件{first_member_info.filename}测试".encode(
                        'utf-8'))
    else:
        # zipfile的存档中成员的自动化命名规则更加合理，即直接以给定的源文件目录的作为存档内成员的根目录
        # 但是tar却并非如此，而是直接按照后续给定的原文件目录的根目录作为存档内成员的根目录
        cmd = f'python -m zipfile -c {tgt_file} {src_root}'
        print("以相对路径指定的源文件目录层级作为成员名字的根目录来进行打包：", cmd)
        subprocess.run(args=cmd, shell=True)

        cmd = f'python -m zipfile -l {tgt_file}'
        print("列出存档中的文件：", cmd)
        subprocess.run(args=cmd, shell=True)

        src_root = os.path.abspath(src_root)
        cmd = f'python -m zipfile -c {tgt_file} {src_root}'
        print("以绝对路径指定的源文件目录层级作为成员名字的根目录来进行打包：", cmd)
        subprocess.run(args=cmd, shell=True)

        cmd = f'python -m zipfile -l {tgt_file}'
        print("列出存档中的文件：", cmd)
        subprocess.run(args=cmd, shell=True)

        cmd = f'python -m zipfile -e {tgt_file} {new_src_root1}'
        print("解压存档到指定目录：", cmd)
        subprocess.run(args=cmd, shell=True)

        cmd = f'python -m zipfile -t {tgt_file}'
        print("测试存档是否正常：", cmd)
        subprocess.run(args=cmd, shell=True)


def tar_compression(start_root, src_root, tgt_file, new_src_root1, new_src_root2, use_cmd):
    """
    tar: https://docs.python.org/3/library/tarfile.html

    tarfile 模块可以用来读写 tar 归档，包括使用 gzip, bz2 和 lzma 压缩的归档。
    请使用 zipfile 模块来读写 .zip 文件，或者使用 shutil 的高层级函数。
    """
    if not use_cmd:
        src_files = [os.path.join(src_root, x) for x in sorted(os.listdir(src_root))]
        with tarfile.open(tgt_file, "w") as tar:
            for src_file in src_files:
                # 将文件 name 添加到归档。
                # name 可以为任意类型的文件（目录、fifo、符号链接等等）。
                # 如果给出 arcname 则它将为归档中的文件指定一个替代名称。
                # 默认情况下会递归地添加目录。这可以通过将 recursive 设为 False 来避免。递归操作会按排序顺序添加条目。
                tar.add(src_file, arcname=os.path.relpath(src_file, start=start_root),
                        recursive=True)

        with tarfile.open(tgt_file, "r") as tar:
            print(f"tar.list 存档中的文件列表：")
            tar.list(verbose=True)
            # getnames内部已经执行了一遍next()，所以该上下文中不能再用next获得有效信息了
            member_names = tar.getnames()
            print(f"存档中的成员名字：{member_names}")
            first_member_name = member_names[0]
            first_member = tar.getmember(name=first_member_name)
            assert first_member == tar.getmember(name=first_member_name)

        with tarfile.open(tgt_file, "r") as tar:
            print(f"tar.next 存档中的文件列表：")
            next_member = tar.next()
            while next_member is not None:
                print(next_member.name)
                next_member = tar.next()

        # 绝不要未经预先检验就从不可靠的源中提取归档文件。
        # 这样有可能在 path 之外创建文件，例如某些成员具有以 "/" 开始的绝对路径文件名或带有两个点号 ".." 的文件名。
        with tarfile.open(tgt_file, 'r') as tar:
            members = tar.getmembers()
            for next_member in members:
                print(f"正在提取：{next_member.name} {next_member.size}")
                if next_member.isfile() or next_member.issym() or next_member.islnk():
                    member_fileobj: io.BufferedReader = tar.extractfile(member=next_member)
                    with open(os.path.join(new_src_root1, next_member.name), mode='wb') as f:
                        f.write(member_fileobj.read())
                else:
                    # extract() 方法不会处理某些提取问题。 在大多数情况下你应当考虑使用 extractall() 方法。
                    tar.extract(member=next_member, path=new_src_root1)

        with tarfile.open(tgt_file, "r") as tar:
            # 如果给定了可选的 members，则它必须为 getmembers() 所返回的列表的一个子集。
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=new_src_root2, members="None")
    else:
        # 在直接使用终端指令处理时，tar存档中存放的成员名字均是实际基于存档文件名后跟的目录进行扩展的
        # 所以使用终端指令最好就是直接在要打包的目录或者文件处执行打包。
        # 如果想要将存档生成到他处，可以对存档路径进行修改。
        tgt_file = os.path.abspath(tgt_file)
        new_src_root1 = os.path.abspath(new_src_root1)

        cmd = f'python -m tarfile -c {tgt_file} {new_src_root1}'
        print("打包绝对路径上的文件、目录到绝对路径上的存档：", cmd)
        subprocess.run(cmd, shell=True)

        cmd = f'python -m tarfile -l {tgt_file}'
        print("列出绝对路径上的存档里的成员：", cmd)
        subprocess.run(cmd, shell=True)

        cmd = f'cd {os.path.dirname(new_src_root1)} && python -m tarfile -c {tgt_file} {new_src_root1}'
        print("在源文件目录的同级目录下打包绝对路径上的源文件目录到绝对路径上的存档：", cmd)
        subprocess.run(cmd, shell=True)

        cmd = f'python -m tarfile -l {tgt_file}'
        print("列出绝对路径上的存档里的成员：", cmd)
        subprocess.run(cmd, shell=True)

        cmd = f'cd {os.path.dirname(new_src_root1)} && python -m tarfile -c {tgt_file} {os.path.basename(new_src_root1)}'
        print("在源文件目录的同级目录下打包相对路径上的源文件目录到绝对路径上的存档：", cmd)
        subprocess.run(cmd, shell=True)

        cmd = f'python -m tarfile -l {tgt_file}'
        print("列出绝对路径上的存档里的成员：", cmd)
        subprocess.run(cmd, shell=True)

        cmd = f'python -m tarfile -e {tgt_file} {new_src_root2}'
        print("提取绝对路径上的存档到绝对路径上的目标文件夹：", cmd)
        subprocess.run(cmd, shell=True)


if __name__ == '__main__':
    zlib_compression()
    zlib_compression(src_file='./data/src/测试/夕小瑶AI全栈手册/目录截图.jpg',
                     tgt_file='./data/tgt/目录截图.zlib',
                     new_src_file="./data/tgt/目录截图.jpg",
                     is_stream=True)
    gzip_compression()
    gzip_compression(src_file='./data/src/测试/夕小瑶AI全栈手册/0.编程基础/7款优秀Vim插件帮你打造完美IDE.pdf',
                     tgt_file='./data/tgt/7款优秀Vim插件帮你打造完美IDE.pdf.gz',
                     new_src_file="./data/tgt/7款优秀Vim插件帮你打造完美IDE.pdf",
                     use_copy=True,
                     is_stream=False)
    gzip_compression(src_file='./data/src/测试/夕小瑶AI全栈手册/0.编程基础/7款优秀Vim插件帮你打造完美IDE.pdf',
                     tgt_file='./data/tgt/7款优秀Vim插件帮你打造完美IDE.pdf.gz',
                     new_src_file="./data/tgt/7款优秀Vim插件帮你打造完美IDE.pdf",
                     use_copy=False,
                     is_stream=True)
    bzip2_compression()
    bzip2_compression(src_file='./data/src/测试/夕小瑶AI全栈手册/0.编程基础/7款优秀Vim插件帮你打造完美IDE.pdf',
                      tgt_file='./data/tgt/7款优秀Vim插件帮你打造完美IDE.pdf.bz2',
                      new_src_file="./data/tgt/7款优秀Vim插件帮你打造完美IDE.pdf",
                      is_stream=False)
    bzip2_compression(src_file='./data/src/测试/夕小瑶AI全栈手册/0.编程基础/7款优秀Vim插件帮你打造完美IDE.pdf',
                      tgt_file='./data/tgt/7款优秀Vim插件帮你打造完美IDE.pdf.bz2',
                      new_src_file="./data/tgt/7款优秀Vim插件帮你打造完美IDE.pdf",
                      is_stream=True)
    lzma_compression()
    lzma_compression(src_file='./data/src/测试/夕小瑶AI全栈手册/0.编程基础/7款优秀Vim插件帮你打造完美IDE.pdf',
                     tgt_file='./data/tgt/7款优秀Vim插件帮你打造完美IDE.pdf.xz',
                     new_src_file="./data/tgt/7款优秀Vim插件帮你打造完美IDE.pdf",
                     is_stream=False)
    lzma_compression(src_file='./data/src/测试/夕小瑶AI全栈手册/0.编程基础/7款优秀Vim插件帮你打造完美IDE.pdf',
                     tgt_file='./data/tgt/7款优秀Vim插件帮你打造完美IDE.pdf.xz',
                     new_src_file="./data/tgt/7款优秀Vim插件帮你打造完美IDE.pdf",
                     is_stream=True)
    zip_compression(
        src_root='./data/src/a',
        tgt_file='./data/tgt/a.zip',
        new_src_root1='./data/tgt/a',
        new_src_root2='./data/tgt/b',
        use_cmd=False
    )
    zip_compression(
        src_root='./data/src/a',
        tgt_file='./data/tgt/a.zip',
        new_src_root1='./data/tgt/a',
        new_src_root2='./data/tgt/b',
        use_cmd=True
    )
    tar_compression(
        start_root='./data/src',
        src_root='./data/src/a',
        tgt_file='./data/tgt/a.tar',
        new_src_root1='./data/tgt/a',
        new_src_root2='./data/tgt/b',
        use_cmd=False,
    )
    tar_compression(
        start_root='./data/src',
        src_root='./data/src/a',
        tgt_file='./data/tgt/a.tar',
        new_src_root1='./data/tgt/a',
        new_src_root2='./data/tgt/b',
        use_cmd=True,
    )
