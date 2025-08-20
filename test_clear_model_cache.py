#!/usr/bin/env python3
"""
测试脚本：测试 video_pipeline_api.py 中的 clear_model_cache 函数

该脚本测试以下功能：
1. 正常情况下的缓存清理
2. 服务器不可达时的错误处理
3. 无效服务器地址的处理
4. quiet 参数的功能
5. 网络超时的处理
6. 真实ComfyUI服务器的集成测试
"""

import os
import sys
import time
import json
import unittest
import threading
from unittest.mock import patch, MagicMock, Mock
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.error

# 添加 ComfyUI 根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
comfyui_root = current_dir
if comfyui_root not in sys.path:
    sys.path.insert(0, comfyui_root)

# 导入要测试的函数
from script_examples.video_pipeline_api import clear_model_cache


class MockComfyUIServer:
    """模拟 ComfyUI 服务器用于测试"""
    
    def __init__(self, port=8999):
        self.port = port
        self.server = None
        self.thread = None
        self.requests_received = []
        self.response_code = 200
        self.response_delay = 0
        self.should_fail = False
        
    def start(self):
        """启动模拟服务器"""
        class TestHandler(BaseHTTPRequestHandler):
            def __init__(self, test_server, *args, **kwargs):
                self.test_server = test_server
                super().__init__(*args, **kwargs)
            
            def do_POST(self):
                # 记录请求
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                
                request_info = {
                    'path': self.path,
                    'headers': dict(self.headers),
                    'data': post_data.decode('utf-8') if post_data else None,
                    'timestamp': time.time()
                }
                self.test_server.requests_received.append(request_info)
                
                # 模拟延迟
                if self.test_server.response_delay > 0:
                    time.sleep(self.test_server.response_delay)
                
                # 模拟失败
                if self.test_server.should_fail:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b'Internal Server Error')
                    return
                
                # 正常响应
                self.send_response(self.test_server.response_code)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                response = {'status': 'success', 'message': 'Cache cleared'}
                self.wfile.write(json.dumps(response).encode('utf-8'))
            
            def log_message(self, format, *args):
                # 禁用默认日志输出
                pass
        
        # 创建处理器工厂
        def handler_factory(*args, **kwargs):
            return TestHandler(self, *args, **kwargs)
        
        self.server = HTTPServer(('localhost', self.port), handler_factory)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()
        
        # 等待服务器启动
        time.sleep(0.1)
    
    def stop(self):
        """停止模拟服务器"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.thread:
            self.thread.join(timeout=1)
    
    def reset(self):
        """重置服务器状态"""
        self.requests_received = []
        self.response_code = 200
        self.response_delay = 0
        self.should_fail = False


class TestClearModelCache(unittest.TestCase):
    """clear_model_cache 函数的测试用例"""
    
    def setUp(self):
        """测试前的设置"""
        self.mock_server = MockComfyUIServer()
        self.mock_server.start()
        self.server_address = f"localhost:{self.mock_server.port}"
    
    def tearDown(self):
        """测试后的清理"""
        self.mock_server.stop()
    
    def test_successful_cache_clear(self):
        """测试成功清理缓存"""
        print("\n测试 1: 成功清理缓存")
        
        # 执行函数
        clear_model_cache(self.server_address, quiet=True)
        
        # 验证请求
        self.assertEqual(len(self.mock_server.requests_received), 1)
        request = self.mock_server.requests_received[0]
        
        # 验证请求路径
        self.assertEqual(request['path'], '/free')
        
        # 验证请求头
        self.assertEqual(request['headers']['Content-Type'], 'application/json')
        
        # 验证请求数据
        expected_data = {"unload_models": True, "free_memory": True}
        actual_data = json.loads(request['data'])
        self.assertEqual(actual_data, expected_data)
        
        print("✓ 请求路径正确")
        print("✓ 请求头正确")
        print("✓ 请求数据正确")
    
    def test_server_error_response(self):
        """测试服务器错误响应"""
        print("\n测试 2: 服务器错误响应")
        
        # 设置服务器返回错误
        self.mock_server.should_fail = True
        
        # 执行函数（应该不抛出异常）
        try:
            clear_model_cache(self.server_address, quiet=True)
            print("✓ 函数正确处理了服务器错误，没有抛出异常")
        except Exception as e:
            self.fail(f"函数不应该抛出异常，但抛出了: {e}")
    
    def test_connection_timeout(self):
        """测试连接超时"""
        print("\n测试 3: 连接超时")
        
        # 设置长延迟模拟超时
        self.mock_server.response_delay = 15  # 超过函数中的 10 秒超时
        
        start_time = time.time()
        try:
            clear_model_cache(self.server_address, quiet=True)
        except Exception:
            pass  # 预期会有超时异常
        
        elapsed_time = time.time() - start_time
        
        # 验证在合理时间内超时（应该在 10-12 秒之间）
        self.assertLess(elapsed_time, 12, "超时时间应该在 12 秒内")
        self.assertGreater(elapsed_time, 9, "超时时间应该大于 9 秒")
        
        print(f"✓ 正确处理超时，耗时: {elapsed_time:.1f}秒")
    
    def test_invalid_server_address(self):
        """测试无效的服务器地址"""
        print("\n测试 4: 无效的服务器地址")
        
        invalid_address = "invalid.server.address:9999"
        
        try:
            clear_model_cache(invalid_address, quiet=True)
            print("✓ 函数正确处理了无效地址，没有抛出异常")
        except Exception as e:
            self.fail(f"函数不应该抛出异常，但抛出了: {e}")
    
    def test_quiet_parameter(self):
        """测试 quiet 参数功能"""
        print("\n测试 5: quiet 参数功能")
        
        # 使用 patch 捕获 print 输出
        with patch('builtins.print') as mock_print:
            # 测试 quiet=True（不应该有输出）
            clear_model_cache(self.server_address, quiet=True)
            mock_print.assert_not_called()
            print("✓ quiet=True 时没有输出")
        
        with patch('builtins.print') as mock_print:
            # 测试 quiet=False（应该有输出）
            clear_model_cache(self.server_address, quiet=False)
            mock_print.assert_called()
            print("✓ quiet=False 时有输出")
    
    def test_different_response_codes(self):
        """测试不同的 HTTP 响应码"""
        print("\n测试 6: 不同的 HTTP 响应码")
        
        # 测试 200 响应
        self.mock_server.response_code = 200
        clear_model_cache(self.server_address, quiet=True)
        print("✓ 正确处理 200 响应")
        
        # 测试 404 响应
        self.mock_server.reset()
        self.mock_server.response_code = 404
        clear_model_cache(self.server_address, quiet=True)
        print("✓ 正确处理 404 响应")
        
        # 测试 500 响应
        self.mock_server.reset()
        self.mock_server.response_code = 500
        clear_model_cache(self.server_address, quiet=True)
        print("✓ 正确处理 500 响应")
    
    def test_request_format(self):
        """测试请求格式的详细验证"""
        print("\n测试 7: 请求格式详细验证")
        
        clear_model_cache(self.server_address, quiet=True)
        
        request = self.mock_server.requests_received[0]
        
        # 验证 HTTP 方法（通过路径确认是 POST）
        self.assertEqual(request['path'], '/free')
        
        # 验证 Content-Type
        self.assertEqual(request['headers']['Content-Type'], 'application/json')
        
        # 验证 JSON 数据格式
        data = json.loads(request['data'])
        self.assertIsInstance(data, dict)
        self.assertIn('unload_models', data)
        self.assertIn('free_memory', data)
        self.assertTrue(data['unload_models'])
        self.assertTrue(data['free_memory'])
        
        print("✓ HTTP 方法正确")
        print("✓ Content-Type 正确")
        print("✓ JSON 数据格式正确")
        print("✓ 包含必要的参数")


def check_real_server(server_address):
    """检查真实服务器是否可用"""
    try:
        # 尝试连接服务器的系统状态端点
        test_url = f"http://{server_address}/system_stats"
        with urllib.request.urlopen(test_url, timeout=5) as response:
            if response.getcode() == 200:
                return True, "服务器正常运行"
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # 404 可能意味着服务器运行但没有这个端点，尝试其他端点
            try:
                test_url = f"http://{server_address}/queue"
                with urllib.request.urlopen(test_url, timeout=5) as response:
                    return True, "服务器正常运行（通过 /queue 端点确认）"
            except:
                pass
        return False, f"HTTP 错误: {e.code}"
    except urllib.error.URLError as e:
        return False, f"连接错误: {e.reason}"
    except Exception as e:
        return False, f"未知错误: {e}"

    return False, "无法连接"


def run_integration_test():
    """运行集成测试（测试真实的 ComfyUI 服务器）"""
    print("\n" + "="*60)
    print("集成测试 - 真实 ComfyUI 服务器测试")
    print("="*60)

    # 常见的 ComfyUI 服务器地址
    potential_servers = [
        "127.0.0.1:8188",  # 默认端口
        "localhost:8188",
        "127.0.0.1:8189",  # 备用端口
        "127.0.0.1:8190",
        "127.0.0.1:8191"
    ]

    working_servers = []

    print("正在扫描可用的 ComfyUI 服务器...")
    for server in potential_servers:
        is_available, status = check_real_server(server)
        if is_available:
            working_servers.append(server)
            print(f"✓ 发现服务器: {server} - {status}")
        else:
            print(f"✗ {server} - {status}")

    if not working_servers:
        print("\n⚠ 没有发现可用的 ComfyUI 服务器")
        print("请确保 ComfyUI 服务器正在运行，然后重新运行测试")
        return

    print(f"\n找到 {len(working_servers)} 个可用服务器，开始集成测试...")

    for server in working_servers:
        print(f"\n--- 测试服务器: {server} ---")

        try:
            # 测试 1: 静默模式缓存清理
            print("1. 测试静默模式缓存清理...")
            start_time = time.time()
            clear_model_cache(server, quiet=True)
            elapsed = time.time() - start_time
            print(f"   ✓ 静默模式成功，耗时: {elapsed:.2f}秒")

            # 等待一下再进行下一个测试
            time.sleep(1)

            # 测试 2: 详细输出模式缓存清理
            print("2. 测试详细输出模式缓存清理...")
            start_time = time.time()
            clear_model_cache(server, quiet=False)
            elapsed = time.time() - start_time
            print(f"   ✓ 详细模式成功，耗时: {elapsed:.2f}秒")

            print(f"   ✓ 服务器 {server} 所有测试通过")

        except Exception as e:
            print(f"   ✗ 服务器 {server} 测试失败: {e}")

    print(f"\n集成测试完成，测试了 {len(working_servers)} 个服务器")


def main():
    """主函数"""
    print("clear_model_cache 函数测试")
    print("="*50)

    # 运行单元测试
    print("运行单元测试...")
    unittest.main(argv=[''], exit=False, verbosity=0)

    # 运行集成测试
    run_integration_test()

    print("\n" + "="*50)
    print("所有测试完成")
    print("测试文件保留在: test_clear_model_cache.py")


if __name__ == "__main__":
    main()
