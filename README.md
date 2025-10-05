# natural-language-processing

📘 This 



![#gold](https://placehold.co/15x15/gold/gold.svg) **Domain Layer**

![#red](https://placehold.co/15x15/red/red.svg) **Application Layer**

![#green](https://placehold.co/15x15/green/green.svg) **Infrastructure Layer**

![#blue](https://placehold.co/15x15/blue/blue.svg) **Presentation Layer**

![#gray](https://placehold.co/15x15/gray/gray.svg) **External Layer**

> "A computer program is a detailed description of the **policy** by which inputs are transformed into outputs."
>
> — Robert Martin

> [!IMPORTANT]
> - A
> B
> - `A -> B` 


> [!NOTE]
> aaa
> bbb
> ccc

> [!WARNING]
> - aaa
> bbb



<details>
  <summary> Application Controller - Interactor</summary>

  <p align="center">
  <img src="docs/pic1.png" alt="picture description" />
  <br><em>Figure 7: Application Controller - Interactor</em>
  </p>

- bla bla bla....

</details>

## Structure

```
.
├── config/...                                   # configuration files and scripts, includes Docker
├── Makefile                                     # shortcuts for setup and common tasks
├── scripts/...                                  # helper scripts
├── pyproject.toml                               # tooling and environment config (uv)
├── ...
└── src/
    └── app/
        ├── domain/                              # domain layer
        │   ├── services/...                     # domain layer services
        │   ├── entities/...                     # entities (have identity)
        │   │   ├── base.py                      # base declarations
        │   │   └── ...                          # concrete entities
        │   ├── value_objects/...                # value objects (no identity)
        │   │   ├── base.py                      # base declarations
        │   │   └── ...                          # concrete value objects
        │   └── ...                              # ports, enums, exceptions, etc.
        │
        ├── presentation/...                     # presentation layer
        │   └── http/                            # http interface
        │       ├── auth/...                     # web auth logic
        │       ├── controllers/...              # controllers and routers
        │       └── errors/...                   # error handling helpers
        │  
        └── run.py                               # app entry point
```


1. Yaml

```yaml
services:
  app:
    # ...
    environment:
      APP_ENV: ${APP_ENV}
```

2. Shell

```shell
export APP_ENV=local
# export APP_ENV=dev
# export APP_ENV=prod
```

  

[^1]: Session and token share the same expiry time, avoiding database reads if the token is expired.
This scheme of using JWT **is not** related to OAuth 2.0 and is a custom micro-optimization.
