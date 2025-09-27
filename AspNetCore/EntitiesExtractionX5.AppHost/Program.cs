var builder = DistributedApplication.CreateBuilder(args);

// var redis = builder.AddRedis("cache")
//     .WithDataVolume()
//     .WithRedisInsight()
//     .WithLifetime(ContainerLifetime.Persistent);

builder.AddProject<Projects.Api>("api")/*.WithReference(redis)*/;

builder.Build().Run();
