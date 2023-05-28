struct Encoder{M, R, P}
    model::M
    re::R
    p::P

    function Encoder(model; p = nothing)
        _p, re = Flux.destructure(model)
        if p === nothing
            p = _p
        end
        return new{typeof(model), typeof(re), typeof(p)}(model, re, p)
    end
end

Flux.trainable(encoder::Encoder) = (encoder.p,)

(encoder::Encoder)(x, p = encoder.p) = encoder.model(x)



struct Decoder{M, R, P}
    model::M
    re::R
    p::P

    function Decoder(model; p = nothing)
        _p, re = Flux.destructure(model)
        if p === nothing
            p = _p
        end
        return new{typeof(model), typeof(re), typeof(p)}(model, re, p)
    end
end

Flux.trainable(decoder::Decoder) = (decoder.p,)

(decoder::Decoder)(x, p = decoder.p) = decoder.model(x)


