# Cursor 用户数据迁移脚本

将 Cursor 用户数据迁移到 `D:\Cursor\UserData`，通过目录联接保持原路径可用，无需修改快捷方式。

## 文件说明

| 文件 | 用途 |
|------|------|
| `migrate-cursor.bat` | 双击执行迁移（自动请求管理员权限） |
| `migrate-cursor.ps1` | 迁移 PowerShell 脚本 |
| `rollback-cursor.bat` | 双击执行回滚 |
| `rollback-cursor.ps1` | 回滚 PowerShell 脚本 |

## 安装到 D:\Cursor

将本目录下所有文件复制到 `D:\Cursor\`：

```powershell
New-Item -ItemType Directory -Path 'D:\Cursor' -Force
Copy-Item -Path '.\*' -Destination 'D:\Cursor\' -Recurse -Force
```

## 使用方法

1. **完全退出 Cursor**（托盘 → 退出，任务管理器确认无残留进程）
2. 双击 `D:\Cursor\migrate-cursor.bat`
3. 迁移完成后重新启动 Cursor

## 迁移范围

- `%APPDATA%\Cursor` → `D:\Cursor\UserData\AppData\Roaming`
- `%LOCALAPPDATA%\Cursor` → `D:\Cursor\UserData\AppData\Local`
- `%USERPROFILE%\.cursor` → `D:\Cursor\UserData\UserProfile\.cursor`

## 回滚

双击 `D:\Cursor\rollback-cursor.bat` 即可还原。

## 验证

```powershell
cmd /c dir /AL "%APPDATA%\Cursor"
```

输出含 `<JUNCTION>` 且指向 `D:\Cursor\UserData\...` 即为成功。
