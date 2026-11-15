# 设置 ▸ 通用 — 外观、文字大小、启动与命令行（src/general.rs），简体中文。

general-appearance = 外观

general-day-night = 昼夜
general-day-night-auto = 自动
general-day-night-system = 系统
general-day-night-day = 白天
general-day-night-night = 夜间
general-day-night-choice = 昼夜：{ $choice }
general-day-night-note = “系统”跟随 macOS。“自动”跟随太阳 — 日出到日落之间为白天。

general-time-of-day = 时段
general-on = 开
general-off = 关
general-time-of-day-choice = 时段：{ $choice }
general-time-of-day-note = 跟随阳光的变化 — 日出前后偏冷，正午和午夜中性，日落前后偏暖。日出和日落时间根据你的时区估算。

general-light = 光线
general-light-cool = 冷
general-light-neutral = 中性
general-light-warm = 暖
general-light-choice = 光线：{ $choice }
general-light-note = 为配色指定固定的光线特征。

general-text-size = 文字大小
general-text-size-smaller-name = 缩小文字，当前 { $percent }
general-text-size-larger-name = 放大文字，当前 { $percent }
general-text-size-reset = 重置
general-text-size-reset-name = 将文字大小重置为 100%
general-text-size-note = 缩放所有文字。也可在 View 菜单中使用 — Actual Size、Zoom In、Zoom Out。

general-startup = 启动
general-open-at-login = 登录时打开
general-login-item-switch = 登录时启动 Eidola
general-login-item-name = 登录时打开 Eidola
general-login-item-off = 在你打开之前，Eidola 不会出现。
general-login-item-on = Eidola 随你的会话一起打开。
general-login-item-needs-approval = 已开启，但 macOS 正在等你确认 — 请在“系统设置 → 通用 → 登录项”中允许 Eidola。
general-login-item-unsupported = 不可用 — 需要 macOS 13 或更高版本，以及已安装并签名的 App。
general-login-item-unsupported-linux = 此处不可用 — 在 Linux 上，`eidola service install` 会启用用户服务。
general-login-item-too-old = 登录时打开需要 macOS 13 或更高版本 — 这台 Mac 的系统较旧。
general-login-item-not-here = 此平台不支持登录时打开。

general-command-line = 命令行
general-cli-field = 终端命令
general-cli-not-installed = 未安装。安装后会在 { $dir } 中放置一个链接到此份 Eidola 的 `eidola` 命令，可以在“终端”中运行。
general-cli-needs-admin = macOS 会要求输入管理员密码。
general-cli-installed = 已安装 — 在“终端”中运行 `eidola` 即运行此份 Eidola，经由 { $path }。
general-cli-dangling = { $path } 指向一份已不存在的 Eidola（{ $target }）— App 可能被移动过。更新链接以使用此份 Eidola。
general-cli-other-copy = { $path } 运行的是另一份 Eidola，位于 { $target }。更新链接以使用此份 Eidola。
general-cli-occupied-file = { $path } 已经是一个并非 Eidola 放置的文件，Eidola 不会替换它。如果不再需要，请自行移动或删除，然后再安装。
general-cli-occupied-directory = { $path } 是一个文件夹，Eidola 不会在此放置命令。如果不再需要，请自行移动或删除，然后再安装。
general-cli-occupied-link = { $path } 是指向 { $target } 的链接，它不是 Eidola，Eidola 不会替换它。如果不再需要，请自行移动或删除，然后再安装。
general-cli-occupied-other = { $path } 已被并非 Eidola 放置的内容占用，Eidola 不会替换它。
general-cli-unreadable = 无法检查 { $path } — { $reason }
general-cli-not-bundled = 仅当 Eidola 作为已安装的 App 运行时可用。
general-cli-translocated = macOS 正从临时位置运行此份 Eidola，指向它的链接将会失效。请将 Eidola 移到“应用程序”文件夹，从那里打开，然后再安装。
general-cli-missing = 此份 Eidola 中没有命令行工具。
general-cli-install = 安装
general-cli-install-admin = 安装…
general-cli-install-name = 在 { $dir } 中安装 eidola 命令
general-cli-update = 更新链接
general-cli-update-admin = 更新链接…
general-cli-update-name = 让 { $path } 指向此份 Eidola
general-cli-installing = 正在安装…
general-cli-waiting = 正在等待管理员密码…
general-cli-declined = 未做任何更改。若要自行安装，请在“终端”中运行：
general-cli-failed = 无法安装命令 — { $reason }
general-cli-manual = 若要自行安装，请在“终端”中运行：
general-cli-copy = 拷贝
general-cli-copy-name = 拷贝安装命令
