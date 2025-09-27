FROM mcr.microsoft.com/dotnet/aspnet:9.0 AS base
USER $APP_UID
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
ARG BUILD_CONFIGURATION=Release
WORKDIR /src
COPY ["Api/Api.csproj", "Api/"]
RUN dotnet restore "Api/Api.csproj"
COPY . .
WORKDIR "/src/Api"
RUN dotnet build "./Api.csproj" -c $BUILD_CONFIGURATION -o /app/build

FROM build AS publish
ARG BUILD_CONFIGURATION=Release
RUN dotnet publish "./Api.csproj" -c $BUILD_CONFIGURATION -o /app/publish /p:UseAppHost=false
RUN dotnet tool install --global CSnakes.Stage
ENV PATH="/root/.dotnet/tools:${PATH}"
RUN setup-python --python 3.12 --verbose --venv /app/venv --pip-requirements /src/Api/python/requirements.txt 

FROM base AS final
WORKDIR /app
USER root 
COPY --from=publish /app/publish .
COPY --from=publish /root/.config/CSnakes /home/app/.config/CSnakes
COPY --from=publish /app/venv /app/venv
ENTRYPOINT ["dotnet", "Api.dll"]
